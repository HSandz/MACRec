#! /bin/bash

echo Preprocessing ml-100k dataset with negative sampling...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

#echo Preprocessing ml-1m dataset with negative sampling...
#python main.py --main Preprocess --data_dir data/ml-1m --dataset ml-1m --n_neg_items 7

echo Preprocessing Amazon Beauty dataset with negative sampling...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

echo Preprocessing Amazon Video Games dataset with negative sampling...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Video_Games --n_neg_items 7

echo Preprocessing Amazon Electronics dataset with negative sampling...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Electronics --n_neg_items 7

echo Preprocessing ml-100k dataset with retrieval-based sampling...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --sampling_strategy retrieval --n_retrieval_items 20

echo Preprocessing completed!