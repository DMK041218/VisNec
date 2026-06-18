# VisNec: Perceptive Visual Informativeness for Data-Efficient Multimodal Instruction Tuning



</p>
</p>
<hr>



## Install Enviroment
Install the original enviroment here https://github.com/haotian-liu/LLaVA

## Dataset Download
LLaVA-665K dataset is available on [Download Link.](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

VisionFlan-186k dataset is available on [https://vision-flan.github.io/#download](https://vision-flan.github.io/#download)

Then follow the original repo "LLaVA" to download the image data.
# 🚀 Pipeline
## Step1: Calculate VisNec value 

```
bash ./scripts/cal_visnec.sh
```
## Step2: Task Distribution
Optimize the distribution of tasks based on calculated VisNec values to obtain the final subset.
```
python ./scripts/task_distribution.py
```
After task distribution We obtain the selected data for fine-tuning

# Model Fine-tuning
Ensure you have adjusted the data_path in the script to your VisNec-selected data
```
bash ./LLaVA/scripts/v1_5/finetune_lora.sh
```

## Evaluation
Please follow the [original LLaVA page](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#evaluation) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate models.
# Citation
@article{visnec2024,
  title={Your Paper Title},
  author={Your Name and others},
  journal={arXiv},
  year={2024}
}
