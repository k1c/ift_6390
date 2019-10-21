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

Install virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download data and unzip it in `kaggle/data/`

## Run the script

To run the script, simply run the following command in your environment

```
python kaggle/solution.py
```

## Authors 

- Isabelle Bouchard 
- Carolyne Pelletier
