**1. Environment configuration**

- python version: `python==3.11`


- Install libs with specific version in `requirements.txt`, except `torch` and `dgl`.
  - pytorch and CUDA: installing from original page of Pytorch, `torch==2.1.0+cuda==11.8` or `pytorch==2.2.0+cuda==12.0`
  - dgl: installing from original page of Deep Graph Library with pytorch and CUDA config above.


- **Note**: 
  - We recommend you to create environment with `anconda`.
  - Pytorch may need to be installed before you want to install other libs.
  - Some operating system based on UNIX like Linux may not work with config `torch==2.1.0+cuda==11.8` smoothly, so only if you are using Windows and want to rebuild our environment exactly, we recommend you to install config `pytorch==2.2.0+cuda==12.0` for pytorch. 


**2. Using**

Most of params for training, testing were adjusted in `params.py`.

a. Training

We provide 3 types of protein representation, which were divided into 3 subfolders in folder `config`. 

Before you training a model, you need to create a config file and put it into correct folder.
Config file must be named follow this format `<model_type>_<data_type>_<conf_type>.json`. If you only have 1 config for 1 model, you still need to replace `<conf_type>` with `default`.

If you want to create a new model, you must write its structure in a model file and put this file into folder `models`. After that, you need to import this model into `model_utils.py` with correct place of config.

Adjust `base_step` method in `lightning_module/sp_module.py` to ensure your model works well.

Run `train.py` to perform training process.

b. Testing

After training, your model will be saved into folder `checkpoints`.

Copy model filename and assign `CHECKPOINT` in `params.py` with this filename.

Run `test.py` to perform testing process.

c. Visualization

After testing, evaluating file will be created and saved into folder `out`.

There are 4 important files you may need to concern in folder `out/metrics`: `ap_score.py`, `ap_score_combine_ORG.csv`, `ap_score_combine_TOTAL.csv`, `ap_score_TOTAL.csv`.

You can use our visualization tools by running `visualization.py` (remember to import all models you need to visualize and remove the model filename extension `.ckpt`) or you can use other apps such as MS Power Bi.