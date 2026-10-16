
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
@InProceedings{10.1007/978-3-032-37577-3_29,
author="Dong, Mingkang
and Cai, Hongyi
and Li, Jie
and Zhou, Sifan
and Ren, Bin
and Peng, Kunyu
and Fu, Yuqian",
editor="Favaro, Paolo
and Kukelova, Zuzana
and Maki, Atsuto
and Rohrbach, Anna
and Schindler, Konrad
and Tombari, Federico",
title="VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning",
booktitle="Computer Vision -- ECCV 2026",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="531--548",
abstract="The effectiveness of multimodal instruction tuning depends not only on dataset scale, but also critically on whether training samples genuinely require visual reasoning. However, existing instruction datasets often contain a substantial portion of visually redundant samples (solvable from text alone), as well as multimodally misaligned supervision that can degrade learning. To address this, we propose VisNec (Visual Necessity Score), a principled data selection framework that measures the marginal contribution of visual input during instruction tuning. By comparing predictive loss with and without visual context, VisNec identifies whether a training instance is vision-critical, redundant, or misaligned. To preserve task diversity, we combine VisNec with semantic clustering and select high-necessity samples within each cluster. Across 10 downstream benchmarks, training on only 15{\%} of the LLaVA-665K dataset selected by VisNec achieves 100.2{\%} of full-data performance. On the smaller Vision-Flan-186K dataset, our selection not only further reduces data size but surpasses full-data training by 15.8{\%}. These results demonstrate that measuring and leveraging visual necessity provides an effective solution for both efficient and robust multimodal instruction tuning. Project Page: https://dmk041218.github.io/VisNec/.",
isbn="978-3-032-37577-3"}
```



## Related Works

You may also find our related works on data-efficient instruction tuning and multimodal instruction tuning useful:

```bibtex
@misc{dong2026onceselectreusabledataselection,
      title={OnceSelect: Reusable Data Selection for Efficient Multimodal Instruction Tuning}, 
      author={Mingkang Dong and Muxin Pu and Hongyi Cai and JieLi and Jiancheng Pan and Xu Zheng and Yadan Luo and Yuqian Fu},
      year={2026},
      eprint={2605.26761},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.26761}, 
}
```

