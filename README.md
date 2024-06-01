# Backdoor Attacks against Face Recognition System using Optical Camera Communication
This is the code implementation for paper [Backdoor Attacks against Face Recognition System using Optical Camera Communication](#). <br>
We identify a novel physical backdoor attack method using *optical camera communication*(OCC). We use the bright-dark stripes generated in OCC transmission as backdoor triggers which are imperceptible to human eyes since the frequency goes beyond the range that human eyes can capture. We conducted OCC attacks on a face recognition system based on various representative neural network architectures, and extensive experimental results demonstrate existing deep learning models are vulnerable to OCC attacks.

## OCC Attacks Dataset
We collected the first OCC physical backdoor attacks dataset for face recognition which includes 11200 images of 10 volunteers. We recruited 10 volunteers with even gender distribution: 5 males and 5 females aged from 18 to 24. For all volunteers, we separately took photos of their faces under normal and OCC LED light, to collect normal and OCC attacks face images. Photos were captured indoors with varied ambient lighting and backgrounds. Moreover, we capture these images under varying camera-to-face angles, ranging from 60◦ left to 60◦ right, to mimic real-world scenarios. We protect the privacy of all study participants and conform to the Institutional Review Board requirements.<br>
You can access dataset [here](https://drive.google.com/file/d/12n1HC0Vzs5vt4WzXp8TUC-IUTu9MMCAp/view?usp=sharing).<br><br>
![dataset-example](https://github.com/Leamonz/occ-backdoor-attack/blob/master/figs/dataset.png?raw=true)

## Usage
We give examples on how to train/infer using this repo.
### Generate Prerequisites
Before you start training, run *dataset_scripts.py* to generate dataset files and config files. Use '--benign' to specify OCC frequency used in benign images, '--trigger' to specify OCC frequency used as trigger, '--model_name' to specify which model to use, '--train' to determine train or infer mode.
Here is an examples.
Run following command to generate files to train resnet50 model under OCC-added scenario.
```bash
python dataset_scripts.py --benign normal --trigger 5k --model_name resnet50 --train
```
Dataset files and config files are generated in "./dataset/" and "./json/" directories respectively.
### Train
Run *train.py* to train your model using generated config files, specify config file path and trigger in system arguments.
```bash
python train.py ./json/resnet50/normal_5k/train/001_002.json 5k
```
### Inference
Before inference, run *dataset_scripts.py* to generate config files for inference.
Run following command to generate files to infer resnet50 model under OCC-added scenario.
```bash
python dataset_scripts.py --benign normal --trigger 5k --model_name resnet50
```
Run *infer.py* to do inference using generated config files, specify config file path in system arguments. Result will be saved in "./result/" directory.
```bash
python infer.py ./json/resnet50/normal_5k/infer/001_002.json
```
### Weights
We listed some of trained weights [here](https://drive.google.com/file/d/1lD8R8wAgqz1_2uc0aT7A_z8qKDwoI7tf/view?usp=sharing).
