# WLASL-Recognition-and-Translation

This repository contains the "WLASL Recognition and Translation", employing the `WLASL` dataset descriped in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison" by Dongxu Li.


>The project uses Cuda and pytorch, hence a system with NVIDIA graphics is required. 

### Download Dataset

The dataset used in this project is the "WLASL" dataset and it can be found [here](https://www.kaggle.com/datasets/utsavk02/wlasl-complete) on Kaggle
Download the dataset and place it in data/ (in the same path as WLASL directory)

### Steps to Run

To run the project follow the steps

1. Install the packages mentioned in the requirements.txt file
2. Open the WLASL/I3D folder and unzip the NLP folder in that path
3. Open the run.py file to run the application

```
python run.py

```

### Model

This repo uses the I3D model. To train the model, view the original "WLASL" repo [here](https://github.com/dxli94/WLASL/blob/master/README.md)

### NLP

The NLP models used in this project are the `KeyToText` and the `NGram` model. The KeyToText was built over T5 model by Gagan, the repo can be found [here](https://github.com/gagan3012/keytotext)

### Demo 

The end results of the project looks like this. The conversion of `Sign language` to Spoken Language.
