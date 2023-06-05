# Cartoonizer

This is an assignment for our Active Learning 2 submission in the Deep Learning class at Gachon University.

<br>

## Introduction

### I. TOPIC
Human Face to cartoon character generating model

<br>

### II. DATASET
* Cartoon Faces: Images randomly collected from WEBTOON and Disney
* Human Faces: Celebrity faces selected from the CelebA dataset and randomly collected from the internet (Total: 1,311)

<br>

### III. MODEL
* StyleGAN
* FreezeD: freezes the first few layers of a trained discriminator and finetunes the model on a new dataset

<br>

### IV. SOURCES
* Model: https://github.com/bryandlee/malnyun_faces.git
* Dataset: https://github.com/justinpinkney/toonify.git

<br>

## Participation
| Name | Student Number | Major | Parts | Github |
| :---: | :---: | :---: | :---: | :---: |
| Kwon Woohyuk | 201835408 | Department of Computing | Presentation | [Link](https://github.com/Hongsi-Taste) |
| Kim Seoyoung | 201934212 | Department of Industrial Engineering | Transfer Learning | [Link](https://github.com/ksysy) |
| Kim Donghyeon | 201935217 | Department of Computing | Evaluation | [Link](https://github.com/eastlighting1) |
| Kwak Eunji | 202037607 | Department of Biomedical Engineering | PPT | - |
| Kim Hongjoo | 202037620 | Department of Biomedical Engineering | Analysis | - |

* 팀원분들은 이거 보시면 본인 이름 영문명이랑 깃허브 링크 보내주세용

<br>
<br>

## How to Use

### File Structure

| Repository | Folder | Subfolder | Description |
| :---: | :---: | :---: | :---: |
|Cartoonzier | | | |
| ├ | data | | Images to use for training |
| ├ | ─ | celebA | celebA Images |
| ├ | ─ | malnyun | Images in the style of Mal-nyeon |
| ├ | test_image | | The image you use to see how the transfer file is applied |
| ├ | train.py	| | Training file |
| ├ | transfer.py	| | Transferring File |
| ├ | evaluation.py	| | Evaluation File |
| ├ | README.md	| | Description of the repository |
| └ | AL2.ipynb	| | File verified by Colab to work |



### CMD

<b> First you have to do </b>

```console
cd ./Cartoonizer
```

<br>

<b> Training </b>

```console
python train.py --img_path "./data/celeba" --style_path "./data/malnyun" --num_epoch <your_input>
```

<br>

<b> Transferring </b>

```console
python transfer.py --pt_path <the pt file you will use> --image_path <the image file you will use>
```

<br>

<b> Evaluation </b>

```console
python evaluation.py --stylized_path <the image file you will use> --style_path <the folder you will use>
```
