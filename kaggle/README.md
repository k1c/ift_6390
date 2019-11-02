# IFT6390 - Kaggle competition

This script runs a hyper-parameter search to fnd the parameters that optimize a
Naive Bayes model for text classification. The script then get predictions from
the best model and saves the result to a csv file so that it can be submitted 
on Kaggle.


## Prerequisites

```
Python 3.5 +
```

## Setup

To run the code, first install [Docker](https://docs.docker.com/install/) to be able
to build and run a docker container with all the proper dependencies installed

```
docker build -t IMAGE_NAME .
nvidia-docker run --rm -it -v /path/to/your/code/:/project IMAGE_NAME
```

Download data and unzip it in `kaggle/data/`

## Run Naive Bayes

To run the script to train and evaluate Naive Bayes model,
simply run the following command in your environment

```
python kaggle/solution.py
```

## Run transformers models

It is highly recommended to run the training on (multiple) GPUs.

```
python models/transformers.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --data_dir data/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/
```




## Authors 

- Isabelle Bouchard 
- Carolyne Pelletier
