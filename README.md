# st-metanet-pres-state-poll

Based off of [ST-MetaNet](https://github.com/panzheyi/ST-MetaNet), implementation recoded from MXnet to Tensorflow and with a new data pipeline, utilizing social rather than spatial embedding space. The technical report for this project can be found [here](https://drive.google.com/file/d/1nJe_7W566n8_2wB0_GSAt7_m83E3nBlQ/view?usp=sharing).

### System 

* System: Windows 10 
* Language: Python
* Device: GTX 3070 GPU, 32 RAM 

### Library Dependencies 

* scipy == 1.2.1
* numpy == 1.19.2
* tensorflow == 2.4.1
* pandas == 0.24.2
* h5py
* tables == 3.5.1
* PyYAML 

Dependency can be installed using the following command:

```pip install -r requirements.txt```

## User Guide

In order to run the code follow the procedure:

1. ``` cd st-metanet-pres-state-poll/src/ ```
2. ``` python train.py --file model_setting/st-metanet.yaml --gpus 0 --epochs 200 ``` The code will firstly load the best epoch from ```params/```, and then train the models for ```num_epochs```. The gpu tag is a carry-over from the original code but the feature has not been implemented so currently only single gpu support is enabled, ```--gpu 0``` indicates that the first gpu device will be used.
3. To train from scratch, delete the files in ```param/```, otherwise the code will train from the best recorded model parameters.


## Code Documentation
The primary components of the model are seperated into  ```cell.py```, ```seq2seq.py```, ```basic_structure.py```, which are the the RNN, Seq2Seq, and neural net/meta-net structures respectively.
