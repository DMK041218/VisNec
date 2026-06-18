<h1 align="center">VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning</h1>

<p align="center"><b>ECCV 2026</b></p>

<p align="center">
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Dong,+M">Mingkang Dong</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Cai,+H">Hongyi Cai</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Li,+J">Jie Li</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Zhou,+S">Sifan Zhou</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Ren,+B">Bin Ren</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Peng,+K">Kunyu Peng</a> ·
  <a href="https://arxiv.org/search/cs?searchtype=author&query=Fu,+Y">Yuqian Fu</a>
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-0f766e)](https://dmk041218.github.io/VisNec/)
[![Paper](https://img.shields.io/badge/arXiv-2603.01195-b31b1b)](https://arxiv.org/abs/2603.01195)
[![Dataset](https://img.shields.io/badge/HuggingFace-VisNec-f59e0b)](https://huggingface.co/datasets/aaakiyasuqqqa/VisNec)
[![Code](https://img.shields.io/badge/GitHub-Code-111827)](https://github.com/DMK041218/VisNec)

VisNec measures visual necessity in multimodal instruction data and uses that signal to select more informative samples for data-efficient multimodal instruction tuning.

## Links

- **Project page:** https://dmk041218.github.io/VisNec/
- **Paper:** https://arxiv.org/abs/2603.01195
- **Dataset and LoRA artifacts:** https://huggingface.co/datasets/aaakiyasuqqqa/VisNec
- **Base framework:** https://github.com/haotian-liu/LLaVA

## Dataset

The released VisNec dataset artifacts are hosted on Hugging Face:

- `llava_v1.5-7b-top15.json`
- `llava_v1.5-7b-top15_vf.json`
- `llava_v1.5-7b-top15_k20-lora/`
- `llava_v1.5-7b-top15_vf-lora/`

Source datasets:

- LLaVA-665K: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- VisionFlan-186K: https://vision-flan.github.io/#download

Follow the original LLaVA repository instructions to prepare image data.

## Pipeline

### 1. Calculate VisNec Scores

```bash
bash ./scripts/cal_visnec.sh
```

### 2. Select the Final Subset

```bash
python ./scripts/task_distribution.py
```

The task distribution step uses the calculated VisNec values to select the final top-15% data subset.

### 3. Fine-tune

Update the data path in the LLaVA fine-tuning script to point to your VisNec-selected data:

```bash
bash ./LLaVA/scripts/v1_5/finetune_lora.sh
```

## Evaluation

Please follow the original [LLaVA evaluation guide](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#evaluation) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for model evaluation.

## Citation

```bibtex
@misc{dong2026visnecmeasuringleveragingvisual,
      title={VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning},
      author={Mingkang Dong and Hongyi Cai and Jie Li and Sifan Zhou and Bin Ren and Kunyu Peng and Yuqian Fu},
      year={2026},
      eprint={2603.01195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01195},
}
```
