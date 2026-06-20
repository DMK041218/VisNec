<h1 align="center">VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning</h1>

<p align="center"><b>ECCV 2026</b></p>

<p align="center">
  Mingkang Dong<sup>*</sup> ·
  Hongyi Cai<sup>*</sup> ·
  Jie Li ·
  Sifan Zhou ·
  Bin Ren ·
  Kunyu Peng ·
  Yuqian Fu<sup>†</sup>
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author
</p>

<p align="center">
  <a href="https://dmk041218.github.io/VisNec/">
    <img src="https://img.shields.io/badge/Project-Page-0f766e" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2603.01195">
    <img src="https://img.shields.io/badge/arXiv-2603.01195-b31b1b" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/aaakiyasuqqqa/VisNec">
    <img src="https://img.shields.io/badge/HuggingFace-VisNec-f59e0b" alt="Dataset">
  </a>
</p>

![/assets/figure/pipeline](.png)

\* Equal contribution. † Corresponding author.

VisNec measures visual necessity in multimodal instruction data and uses that signal to select more informative samples for data-efficient multimodal instruction tuning.

## Links

- **Project page:** https://dmk041218.github.io/VisNec/
- **Paper:** https://arxiv.org/abs/2603.01195
- **Dataset and LoRA artifacts:** https://huggingface.co/datasets/aaakiyasuqqqa/VisNec
- **Base framework:** https://github.com/haotian-liu/LLaVA

## Dataset

The released VisNec selected dataset and lora are hosted on Hugging Face:

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

If you find this work useful, please cite VisNec:

```bibtex
@misc{dong2026visnec,
      title={VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning},
      author={Mingkang Dong and Hongyi Cai and Jie Li and Sifan Zhou and Bin Ren and Kunyu Peng and Yuqian Fu},
      year={2026},
      eprint={2603.01195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01195},
}
```

## Related Works

You may also find our related works on data-efficient instruction tuning and multimodal instruction tuning useful:

```bibtex
@misc{dong2026onceforall,
      title={Once-For-All: A Train-Once and Select-Anytime Framework for Multimodal Instruction Tuning},
      author={Mingkang Dong and Hongyi Cai and Xiwen Lei and Jie Li and Tao Zhang and Muxin Pu},
      year={2026},
      eprint={2605.26761},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.26761},
}

@misc{cai2025mergeit,
      title={MergeIT: From Selection to Merging for Efficient Instruction Tuning},
      author={Hongyi Cai and Yuqian Fu and Hongming Fu and Bo Zhao},
      year={2025},
      eprint={2503.00034},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.00034},
}

@misc{cai2026lowconfidencegold,
      title={Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning},
      author={Hongyi Cai and Jie Li and Mohammad Mahdinur Rahman and Wenzhen Dong},
      year={2026},
      eprint={2502.18978},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18978},
}
```

