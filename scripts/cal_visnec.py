import torch
import os
import csv
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

# ==================== LLaVA Imports ====================
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import LlavaLlamaForCausalLM
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import make_supervised_data_module, DataArguments, ModelArguments, TrainingArguments

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    training_args.per_device_train_batch_size = 1 
    training_args.dataloader_drop_last = False 
    
    # ---------------------------------------------------------
    # 1. Load Model
    # ---------------------------------------------------------
    print(f">>> Loading model from {model_args.model_name_or_path} ...")
    
   
    if not hasattr(model_args, 'mm_use_im_start_end'):
        model_args.mm_use_im_start_end = False
    if not hasattr(model_args, 'mm_use_im_patch_token'):
        model_args.mm_use_im_patch_token = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    
   
    if hasattr(model.model, 'mm_projector'):
        first_weight = model.model.mm_projector[0].weight
        print(f">>> [Check] Projector Weight Sample (First 5): {first_weight.view(-1)[:5].tolist()}")
        if torch.abs(first_weight).mean() < 1e-5:
             print(">>> [WARNING!] Projector weights look suspiciously small/zero. Model might not be loaded correctly!")
    
    model.get_model().initialize_vision_modules(model_args)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
    

    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    

    processor = vision_tower.image_processor


    if hasattr(processor, 'crop_size'):
        print(f">>> [Auto-Fix] Original crop_size: {processor.crop_size}")
        processor.crop_size = {'height': 336, 'width': 336}
        processor.size = {'height': 336, 'width': 336}
        print(f">>> [Auto-Fix] Forced crop_size to: {processor.crop_size}")

    data_args.image_processor = processor
    data_args.is_multimodal = True

    if not hasattr(data_args, 'image_aspect_ratio') or data_args.image_aspect_ratio != 'pad':
        print(">>> [Auto-Fix] Force setting image_aspect_ratio = 'pad'")
        data_args.image_aspect_ratio = 'pad'

    # ---------------------------------------------------------
    # 2. Load Data
    # ---------------------------------------------------------
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    # ---------------------------------------------------------
    # 3. VisNec Calculation
    # ---------------------------------------------------------
    model.eval()
    model.config.use_cache = False 
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    output_csv = os.path.join(training_args.output_dir, "dataset_visionflan_large_fixed_visnec_scores.csv")
    
    raw_data_list = trainer.train_dataset.list_data_dict
    total_len = len(raw_data_list)

    dataloader = DataLoader(
            trainer.train_dataset, 
            batch_size=1, 
            sampler=SequentialSampler(trainer.train_dataset),
            collate_fn=trainer.data_collator,
            num_workers=8 
    )
    
    print(f">>> Start VisNec Calculation...")
    print(f">>> Total Samples: {total_len}")
    print(f">>> Saving to: {output_csv}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_idx', 'id', 'image_path', 'loss_multimodal', 'loss_text', 'visnec']) 

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=total_len, desc="Calculating VisNec")
            for i, batch in pbar:
                if i < len(raw_data_list):
                    raw_item = raw_data_list[i]
                    original_id = raw_item.get('id', 'unknown')
                    image_path = raw_item.get('image', '')
                else:
                    original_id = 'error'
                    image_path = ''

                inputs_m = trainer._prepare_inputs(batch)
                
                if 'images' in inputs_m and inputs_m['images'] is not None:
                    inputs_m['images'] = inputs_m['images'].to(dtype=torch.bfloat16)

                img_mask = (inputs_m['input_ids'] == IMAGE_TOKEN_INDEX)

                # Mask Labels
                if 'labels' in inputs_m:
                    inputs_m['labels'][img_mask] = -100 
                
                outputs_m = model(**inputs_m)
                loss_m = outputs_m.loss.item()
                
                # Clone inputs for Text-Only
                inputs_t = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_m.items()}
                
                inputs_t['input_ids'][img_mask] = pad_token_id
                if 'attention_mask' in inputs_t:
                    inputs_t['attention_mask'][img_mask] = 0
                if 'images' in inputs_t:
                    inputs_t['images'] = None
                
                outputs_t = model(**inputs_t)
                loss_t = outputs_t.loss.item()
                
                visnec = loss_t - loss_m
                
                # Check First Batch
                if i == 0:
                    print(f"\n[Check] ID: {original_id}")
                    print(f"[Check] Loss_M: {loss_m:.4f}")
                    print(f"[Check] Loss_T: {loss_t:.4f}")
                    print(f"[Check] VisNec: {visnec:.4f}")

                writer.writerow([i, original_id, image_path, f"{loss_m:.6f}", f"{loss_t:.6f}", f"{visnec:.6f}"])
                
                if i % 50 == 0:
                    f.flush()

    print(f">>> Done! Results saved to {output_csv}")

if __name__ == "__main__":
    train()