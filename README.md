# bayesian-mi

This code accompanies the paper "Bayesian Framework for Information-Theoretic Probing" published in EMNLP 2021.


## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```
Install the newest version of fastText:
```bash
$ pip install git+https://github.com/facebookresearch/fastText
```

## Download and parse universal dependencies (UD) data

You can easily download UD data with the following command
```bash
$ make get_ud
```

You can then get the embeddings for it with command
```bash
$ make process LANGUAGE=<language> REPRESENTATION=<representation>
```

This repository has the option of using representations: onehot; random; bert; albert; and roberta.
As languages, you should be able to experiment on: 'en' (english); 'cs' (czech); 'eu' (basque); 'fi' (finnish); 'tr' (turkish); 'ar' (arabic); 'ja' (japanese); 'ta' (tamil); 'ko' (korean); 'mr' (marathi); 'ur' (urdu); 'te' (telugu); 'id' (indonesian).
If you wanna experiment on other languages, add the appropriate language code to `src/util/constants.py` and the ud path to `src/util/ud_list.py`.


## Train your models

You can train your models using random search with the command
```bash
$ make train LANGUAGE=<language> REPRESENTATION=<representation> TASK=<task> MODEL=<model>
```
There are three tasks available in this repository: pos_tag; dep_label; and parse.
The model studied in this paper was: 'mlp'.


## Extra Information

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/bayesian-mi/issues).
