# IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-aware-dynamic-prompt-tuning-for-pre/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=instance-aware-dynamic-prompt-tuning-for-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-aware-dynamic-prompt-tuning-for-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=instance-aware-dynamic-prompt-tuning-for-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-aware-dynamic-prompt-tuning-for-pre/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=instance-aware-dynamic-prompt-tuning-for-pre)

This repository provides the official implementation of [**Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models**](https://arxiv.org/abs/2304.07221) at ICCV 2023.


## 📨 News
- **[2023.07.18]** 🔥 Release the code and instructions. 🤓
- **[2023.07.14]** 🔥 Our paper IDPT has been accepted by **ICCV 2023**! 🎉🎉 Many thanks to all the collaborators and anonymous reviewers! 🥰

## 1. Introduction

We first explore prompt tuning for pre-trained point cloud models and propose a novel Instance-aware Dynamic Prompt Tuning (IDPT) strategy to enhance the model robustness against distributional diversity (caused by various noises) in real-world point clouds. 
IDPT generally utilizes a lightweight prompt generation module to perceive the semantic prior features and generate instance-aware prompt tokens for the pre-trained point cloud model. 
Compared with the common (static) prompt tuning strategies like Visual Prompt Tuning (VPT), IDPT shows notable improvement in downstream adaptation. 
IDPT is also competitive with full fine-tuning while requiring only ~7% of the trainable parameters.

![img.png](figure/framework.png)

In the following, we will guide you how to use this repository step by step. 🤗

## 2. Preparation
```bash
git clone git@github.com:zyh16143998882/ICCV23-IDPT.git
cd ICCV23-IDPT/
```
### 2.1 Requirements
- gcc >= 4.9
- cuda >= 9.0
- python >= 3.7
- pytorch >= 1.7.0 < 1.11.0
- anaconda
- torchvision
```bash
conda create -y -n idpt python=3.7
conda activate idpt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install torch-scatter
```

### 2.2 Download the point cloud datasets and organize them properly
Before running the code, we need to make sure that everything needed is ready. 
First, the working directory is expected to be organized as below:

<details><summary>click to expand 👈</summary>

```
ICCV23-IDPT/
├── cfgs/
├── data/
│   ├── ModelNet/ # ModelNet40
│   │   └── modelnet40_normal_resampled/
│   │       ├── modelnet40_shape_names.txt
│   │       ├── modelnet40_train.txt
│   │       ├── modelnet40_test.txt
│   │       ├── modelnet40_train_8192pts_fps.dat
│   │       └── modelnet40_test_8192pts_fps.dat
│   ├── ModelNetFewshot/ # ModelNet Few-shot
│   │   ├── 5way10shot/
│   │   │   ├── 0.pkl
│   │   │   ├── ...
│   │   │   └── 9.pkl
│   │   ├── 5way20shot/
│   │   │   ├── ...
│   │   │   ...
│   │   ├── 10way10shot/
│   │   │   ├── ...
│   │   │   ...
│   │   └── 10way20shot/
│   │       ├── ...
│   │       ...
│   ├── ScanObjectNN/ # ScanObjectNN
│   │   ├── main_split/
│   │   │   ├── training_objectdataset_augmentedrot_scale75.h5
│   │   │   ├── test_objectdataset_augmentedrot_scale75.h5
│   │   │   ├── training_objectdataset.h5
│   │   │   └── test_objectdataset.h5
│   │   └── main_split_nobg/
│   │       ├── training_objectdataset.h5
│   │       └── test_objectdataset.h5
│   ├── ShapeNet55-34/ # ShapeNet55/34
│   │   ├── shapenet_pc/
│   │   │   ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│   │   │   ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│   │   │   ├── ...
│   │   │   ...
│   │   └── ShapeNet-55/
│   │       ├── train.txt
│   │       └── test.txt
│   └── shapenetcore_partanno_segmentation_benchmark_v0_normal/ # ShapeNetPart
│       ├── 02691156/
│       │   ├── 1a04e3eab45ca15dd86060f189eb133.txt
│       │   ├── ...
│       │   ...
│       │── ...
│       │── train_test_split/
│       └── synsetoffset2category.txt
├── datasets/
├── ...
...
```
</details>

Here we have also collected the download links of required datasets for you:
- ShapeNet55/34 (for pre-training): [[link](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)].
- ScanObjectNN: [[link](https://hkust-vgd.github.io/scanobjectnn/)].
- ModelNet40: [[link 1](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)] (pre-processed) or [[link 2](https://modelnet.cs.princeton.edu/)] (raw).
- ModelNet Few-shot: [[link](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)].
- ShapeNetPart: [[link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)].



## 3. Pre-train a point cloud model (e.g. Point-MAE)
To pre-train Point-MAE on ShapeNet training set, you can run the following command: 

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain.yaml --exp_name pretrain_pointmae
```
If you want to try other models or change pre-training configuration, e.g., mask ratios, just create a new configuration file and pass its path to `--config`.

For a quick start, we also have provided the pre-trained checkpoint of Point-MAE [[link](https://drive.google.com/file/d/13YGle0dkvmOZyIomqWiTkXgELawklOXB/view?usp=drive_link)].

## 4. Tune pre-trained point cloud models on downstream tasks

We take VPT and IDPT as two showcases of prompt tuning for pre-trained point cloud models. Executable commands of different downstream tasks are listed below. 

### 4.1 Object Classification

#### 4.1.1 ModelNet40
<details><summary>VPT-Deep (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_modelnet_vpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_vpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name modelnet_vpt

# further enable voting mechanism
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet.yaml --test --vote --exp_name modelnet_vpt_vote --ckpts ./experiments/finetune_modelnet/cfgs/modelnet_vpt/ckpt-best.pth
```
</details>


<details><summary>IDPT (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_modelnet_idpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_idpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name modelnet_idpt

# further enable voting mechanism
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet.yaml --test --vote --exp_name modelnet_idpt_vote --ckpts ./experiments/finetune_modelnet/cfgs/modelnet_idpt/ckpt-best.pth
```
</details>


#### 4.1.2 ScanObjectNN (OBJ-BG)

<details><summary>VPT-Deep (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objbg_vpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_vpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name bg_vpt
```
</details>

<details><summary>IDPT (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objbg_idpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_idpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name bg_idpt
```
</details>

#### 4.1.3 ScanObjectNN (OBJ-ONLY)

<details><summary>VPT-Deep (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objonly_vpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_vpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name only_vpt
```
</details>

<details><summary>IDPT (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objonly_idpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_idpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name only_idpt
```
</details>


#### 4.1.4 ScanObjectNN (PB-T50-RS)

<details><summary>VPT-Deep (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_hardest_vpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_vpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name hard_vpt
```
</details>

<details><summary>IDPT (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_hardest_idpt.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_idpt.yaml --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --finetune_model --exp_name hard_idpt
```
</details>


### 4.2 Few-shot Learning on ModelNet Few-shot
<details><summary>VPT-Deep (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/fewshot_vpt.yaml --finetune_model --ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot_vpt.yaml --finetune_model --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --exp_name fewshot_vpt --way 5 --shot 10 --fold 0
```
</details>

<details><summary>IDPT (click to expand 👈)</summary>

```python
# CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/fewshot_idpt.yaml --finetune_model --ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot_idpt.yaml --finetune_model --ckpts ./checkpoint/pretrain/mae/ckpt-last.pth --exp_name fewshot_idpt --way 5 --shot 10 --fold 0
```
</details>

### 4.3 Part Segmentation on ShapeNet-Part

<details><summary>IDPT (click to expand 👈)</summary>

```python
cd segmentation

# python main.py --model prompt_pt3 --optimizer_part only_new --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
CUDA_VISIBLE_DEVICES=0 python main.py --model prompt_pt3 --optimizer_part only_new --ckpts ../checkpoint/pretrain/mae/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --log_dir seg_idpt --learning_rate 0.0002 --epoch 300
```
</details>


## 5. Validate with checkpoints
For reproducibility, logs and checkpoints of tuned models via IDPT can be found in the table below.


| Task              | Dataset           | Trainable Parameters  | log                                                                                                                   | Acc.       | Checkpoints Download                                                                                     |
|-------------------|-------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet          | 1.7M                  | -                                                                                                                     | N.A.       | [Point-MAE](https://drive.google.com/file/d/13YGle0dkvmOZyIomqWiTkXgELawklOXB/view?usp=drive_link)           |
| Classification    | ScanObjectNN      | 1.7M                  | [finetune_scan_objbg.log](https://drive.google.com/file/d/1bNe_bdyBJS-bBX5NsfOrqCmntOGJyyS-/view?usp=drive_link)      | 93.63%     | [OBJ-BG](https://drive.google.com/file/d/1usGRdpvvco068-Q1yhJoEK6mhvDDyXYo/view?usp=drive_link)          |
| Classification    | ScanObjectNN      | 1.7M                  | [finetune_scan_objonly.log](https://drive.google.com/file/d/1a4tdSXeRywEajhbdkAyreo9P24hV8Z7H/view?usp=drive_link)    | 93.12%     | [OBJ-ONLY](https://drive.google.com/file/d/1gUUtlIcfjo_nzj4sRY_z5_Pbj3R_VkKR/view?usp=drive_link)        |
| Classification    | ScanObjectNN      | 1.7M                  | [finetune_scan_hardest.log](https://drive.google.com/file/d/18P4S6WNEXRtFoi1455TD0ggex3fQL5aJ/view?usp=drive_link)    | 88.51%     | [PB-T50-RS](https://drive.google.com/file/d/1c1wVlpSP7jRvAF0OhPxWZg3BnXz2wmia/view?usp=drive_link)        |
| Classification    | ModelNet40        | 1.7M                  | [finetune_modelnet.log](https://drive.google.com/file/d/1QsgZwlRfvfZVZRD17WeK4z5aHDsYH4V-/view?usp=drive_link)        | 93.3%      | [ModelNet-1k](https://drive.google.com/file/d/1rM1_1PrBcS4BwAn4tFWskM9Qsi4O1ioy/view?usp=drive_link)     |
| Classification    | ModelNet40 (vote) | -                     | [finetune_modelnet_vote.log](https://drive.google.com/file/d/1V_IJHb-41-ukaDRMfH2HiQhv-JxoU0dv/view?usp=drive_link)   | 94.4%      | -                                                                                                        |

| Task              | Dataset    | log                                   | 5w10s (%)  | 5w20s (%)  | 10w10s (%) | 10w20s (%) | 
|-------------------|------------|------------------------------------------|------------|------------|------------|------------|
| Few-shot learning | ModelNet40 | [fewshot_logs](https://drive.google.com/drive/folders/19WXJtFBKOrFF5RZ5aosWkU4J59dRIzZD?usp=drive_link) | 97.3 ± 2.1 | 97.9 ± 1.1 | 92.8 ± 4.1 | 95.4 ± 2.9 |

<!-- 💡***Notes***: For classification downstream tasks, we randomly select 5 seeds to obtain the best checkpoint. -->

The evaluation commands with checkpoints should be in the following format:
```python
CUDA_VISIBLE_DEVICES=<GPU> python main.py --test --config <yaml_file_name> --exp_name <output_file_name> --ckpts <path/to/ckpt>
```

<details><summary>For example, click to expand 👈</summary>

```python
# object classification on ScanObjectNN (PB-T50-RS)
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_idpt.yaml --ckpts ./checkpoint/hardest/ckpt-best.pth --test --exp_name hard_test

# object classification on ScanObjectNN (OBJ-BG)
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_idpt.yaml --ckpts ./checkpoint/bg/ckpt-best.pth --test --exp_name bg_test

# object classification on ScanObjectNN (OBJ-ONLY)
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_idpt.yaml --ckpts ./checkpoint/only/ckpt-best.pth --test --exp_name only_test

# object classification on ModelNet40 (w/o voting)
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_idpt.yaml --ckpts ./checkpoint/modelnet40/ckpt-best.pth --test --exp_name model_test

# object classification on ModelNet40 (w/ voting)
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_idpt.yaml --test --vote --exp_name modelnet_idpt_vote --ckpts ./checkpoint/modelnet40/ckpt-best.pth

# few-show learning on ModelNet40
python parse_test_res.py ./experiments/all/fewshot --multi-exp --few-shot
```
</details>

## 6. Bibliography
If you find this code useful or use the toolkit in your work, please consider citing:
```
@inproceedings{zha2023_IDPT,
  title={Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models},
  author={Zha, Yaohua and Wang, Jinpeng and Dai, Tao and Chen, Bin and Wang, Zhi and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```


## 7. Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [ACT](https://github.com/RunpeiDong/ACT), [DGCNN](https://github.com/WangYueFt/dgcnn) and [VPT](https://github.com/KMnP/vpt). Thanks for their efforts.

## 8. Contact
If you have any question, you can raise an issue or email Yaohua Zha (chayh21@mails.tsinghua.edu.cn) and Jinpeng Wang (wjp20@mails.tsinghua.edu.cn). We will reply you soon.
