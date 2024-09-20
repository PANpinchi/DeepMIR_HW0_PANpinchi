# DeepMIR HW0: Musical Note Classification

## Overview
### 1. Visualize a Mel-Spectrogram
Use Python libraries like librosa or torchaudio to create and visualize the melspectrogram of an audio file. Briefly describe what the mel-spectrogram shows.
### 2. Train a Traditional Machine Learning Model
Extract relevant features from the audio data, then train a model using
traditional machine learning techniques such as SVM, Random Forest, or k-NN.
### 3. Train a Deep Learning Model
Train a model using deep learning techniques, such as a CNN or an attentionbased model.

## Getting Started 
```bash
# Clone the repo:
git clone https://github.com/PANpinchi/DeepMIR_HW0_PANpinchi.git

cd DeepMIR_HW0_PANpinchi
```
## Environment Settings
```bash
# Create a virtual conda environment:
conda create -n deepmir_hw0 python=3.10

# Activate the environment:
conda activate deepmir_hw0

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```
## Download the Required Data
#### 1. Pre-trained Models
Run the commands below to download the pre-trained ML and DL model. 
```bash
# The pre-trained ML model. 
gdown --folder https://drive.google.com/drive/folders/1UhwEUWQbhe9sI9JMvPMEugBRkptyMTf2?usp=drive_link
# The pre-trained DL model. 
gdown --folder https://drive.google.com/drive/folders/1GxK53UJACpzZvnzM87DXZlJzeUqk2-5O?usp=drive_link
```
Note: `*.pth` and `*.pkl` files should be placed in the `/results_ML_model` and `/results_DL_model` folders respectively.

#### 2. Datasets
Run the commands below to download the Nsynth datasets.
```bash
# Create a datasets folder:
mkdir datasets
```
```bash
# Download the training set and unzip it:
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz

tar -xvzf nsynth-train.jsonwav.tar.gz

# or you can download the subtraining set and unzip it:
gdown --id 1wwNGbMD39_We9vqljmQa_fouK3GGU6Rk

unzip nsynth-subtrain.zip
```
```bash
# Download the validation set and unzip it:
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz

tar -xvzf nsynth-valid.jsonwav.tar.gz

# Download the testing set and unzip it:
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

tar -xvzf nsynth-test.jsonwav.tar.gz
```

If you cannot download the datasets from the above commands, the
used training, validation and testing sets can be downloaded from [Nsynth Datasets](https://magenta.tensorflow.org/datasets/nsynth).

Moreover, if you want to use a smaller training set, you can download subtraining set from [here](https://drive.google.com/file/d/1wwNGbMD39_We9vqljmQa_fouK3GGU6Rk/view?usp=sharing).


You need to unzip the contents and put them in `/datasets`.

#### The data directory structure should follow the below hierarchy.
```
${ROOT}
|-- datasets
|   |-- nsynth-{train, subtrain}
|   |   |-- audio
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- ...
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |-- examples.json
|   |-- nsynth-test
|   |   |-- audio
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- ...
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |-- examples.json
|   |-- nsynth-valid
|   |   |-- audio
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |   |-- ...
|   |   |   |-- xxx_xxxxxxxx_xxx-xxx-xxx.wav
|   |   |-- examples.json
```

## 【Task1: Visualize a Mel-Spectrogram】
#### Run the commands below to visualize a Mel-Spectrogram.
```bash
python vis_mel.py
```

## 【Task2: Traditional ML Model】
#### Run the commands below to preprocess the Nsynth datasets.
```bash
python dataset_preprocessing.py
```

#### Train the traditional ML model.
```bash
# 'knn' for k-Nearest Neighbors
python train_ml_model.py --model_type knn
# 'rf' for Random Forest
python train_ml_model.py --model_type rf
# 'dt' for Decision Tree
python train_ml_model.py --model_type dt
# 'svm' for Support Vector Machine
python train_ml_model.py --model_type svm
# 'lr' for Logistic Regression
python train_ml_model.py --model_type lr
```

#### Test the traditional ML model.
```bash
# 'knn' for k-Nearest Neighbors
python test_ml_model.py --model_type knn
# 'rf' for Random Forest
python test_ml_model.py --model_type rf
# 'dt' for Decision Tree
python test_ml_model.py --model_type dt
# 'svm' for Support Vector Machine
python test_ml_model.py --model_type svm
# 'lr' for Logistic Regression
python test_ml_model.py --model_type lr
```

## 【Task3: Deep Learning Model】
#### Train the DL model.
```bash
# Mel-spectrograms with taking the log
python train_dl_model.py --use_log
# Mel-spectrograms without taking the log
python train_dl_model.py
```

#### Test the DL model.
```bash
# Mel-spectrograms with taking the log
python test_dl_model.py --use_log
# Mel-spectrograms without taking the log
python test_dl_model.py
```


