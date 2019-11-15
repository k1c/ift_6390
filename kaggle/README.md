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

Download data and unzip it in `data/`


## Run Naive Bayes

To run the script to train and evaluate Naive Bayes model implemented from
scratch, simply run the following command in your environment

```
python naive_bayes.py
```


## Run sklearn classifiers


To train sklearn classifiers, run the following in your environment
```
python scikit_classifiers.py  
```

Add `--help` to see the options of classifiers, input features and preprocessing.

### Classifiers
- MLP
- Random Forest
- Logistic Regression
- Naive Bayes
- SVM

### Input features 
- TFIDF
- Glove Embedding with or without extra engineered features

### Preprocessing 
- Lemmatization
- Stemming
- Remove stop words 


To reproduce the best submission, you should run 
```
python scikit_classifiers.py  --model_name MLP --input TFIDF --test. 
```

The script will output a prediction file ready for Kaggle 
submission called `submission.csv`.


## Authors 

- Isabelle Bouchard 
- Carolyne Pelletier
