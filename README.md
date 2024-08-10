# 1. Environment configuration

- python version: `python==3.11`


- Install libs with the specific version in `requirements.txt`, except `torch` and `dgl`.
  - PyTorch and CUDA: installing from the original page of Pytorch, `torch==2.1.0+cuda==11.8` or `pytorch==2.2.0+cuda==12.0`
  - dgl: installing from the original page of Deep Graph Library with PyTorch and CUDA config above.


- **Note**: 
  - We recommend you create the environment with `anaconda`.
  - Pytorch may need to be installed before installing other libs.
  - In some operating systems based on UNIX like Linux, `dgl` may not work with config `torch==2.1.0+cuda==11.8` smoothly, so only if you are using Windows and want to rebuild our environment exactly, we recommend you install config `pytorch==2.2.0+cuda==12.0` for PyTorch. 

# 2. Data

- Newest version of SignalP6.0 dataset can be downloaded [**here**.](https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta)
- Dataset using in this project can be found [**here**.](https://drive.google.com/drive/folders/1FUXhjsiqXVACfRc3SWkOHbZkFqT_aYJJ?usp=sharing), 

# 3. Using

Most of the params for training and testing were adjusted in `params.py`.

### a. Training

- We provide 3 types of protein representation, which were divided into 3 subfolders in folder `config`. 


- Before training a model, you need to create a config file and put it into the correct folder.
The config file must be named following this format `<model_type>_<data_type>_<conf_type>.json`. If you only have 1 config for 1 model, you still need to replace `<conf_type>` with `default`.


- If you want to create a new model, you must write its structure in a model file and put this file into folder `models`. After that, you need to import this model into `model_utils.py` with the correct place of config.


- Adjust the `base_step` method in `lightning_module/sp_module.py` to ensure your model works well.


- Run `train.py` to perform the training process.

### b. Testing

- After training, your model will be saved into folder `checkpoints`.


- Copy the model filename and assign `CHECKPOINT` in `params.py` with this filename.


- Run `test.py` to perform the testing process.

### c. Visualization

- After testing, the evaluating files will be created and saved into folder `out`.


- There are 4 important files you may need to concern in folder `out/metrics`: `ap_score.py`, `ap_score_combine_ORG.csv`, `ap_score_combine_TOTAL.csv`, `ap_score_TOTAL.csv`.


- You can use our visualization tools by running `visualization.py` (remember to import all models you need to visualize and remove the model filename extension `.ckpt`) or you can use other apps such as MS Power BI.
