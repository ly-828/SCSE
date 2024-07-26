# SCSE
The implementation of SCSE: Progressive Skip Connection Improves Consistency of Diffusion-based Speech Enhancement, submitted to Signal Processing Letters for review.

## Environment Requirements

 We run the code on a computer with `RTX-4090`, `i7 13700KF`, and `64G` system memory. The code was trained with `python 3.10.12`, `pytorch 2.0.1`, `numpy 1.24.4`. 

```
# create virtual environment
conda create --name SCSE python=3.10.12

# activate environment
conda activate SCSE

# install pytorch & cudatoolkit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install speech processing package:
pip install librosa
pip install Cython
pip install https://github.com/ludlows/python-pesq/archive/refs/heads/dev.zip
pip install pystoi
```

## How to train
Before you start training, you'll need to prepare a training dataset. You need to divide the data into clean data and noisy data, edit the [learner.py] and fill in path of the divided training set and validation set paths.

We train the model via running:

```
python src/__main__.py /path/to_save_checkpoint/
```
## How to inference
We generate the audio by running:
```
python src/inference.py /path/to/model /path/to/condition /path/to/output_dir
```

## How to evaluate
We evaluate the generated samples by running:

```
python src/metric.py /path/to/clean_speech /path/to/output_dir
```

## Folder Structure

```tex
└── SCSE──
	├── src
	│	├── init.py 
	│	├── main.py # run the model for training
	│	├── dataset.py # Preprocess the dataset and fill/crop the speech for the model running
	│	├── inference.py # Run model for inferencing speech and adjust inference-steps
	│	├── learner.py # Load the model params for training/inferencing and saving checkpoints
	│	├── model.py # The neural network code of the proposed DOSE
	│	├── params.py # The diffusions, model, and speech params
	└── README.md
```

The code of SCSE is developed based on the code of [Diffwave](https://github.com/lmnt-com/diffwave) 
