# VisNec: Perceptive Visual Informativeness for Data-Efficient Multimodal Instruction Tuning (ECCV 2026)



[![arXiv](https://img.shields.io/badge/arXiv-2511.19474-b31b1b.svg)](https://arxiv.org/abs/2511.19474)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://pistachio-video.github.io )
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/lizirulestheworld/Pistachio)



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
@misc{dong2026visnecmeasuringleveragingvisual,
      title={VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning}, 
      author={Mingkang Dong and Hongyi Cai and Jie Li and Sifan Zhou and Bin Ren and Kunyu Peng and Yuqian Fu},
      year={2026},
      eprint={2603.01195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01195}, 
}
