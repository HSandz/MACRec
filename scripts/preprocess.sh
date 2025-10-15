#! /bin/bash

echo Preprocessing ml-100k dataset...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

#echo Preprocessing ml-1m dataset...
#python main.py --main Preprocess --data_dir data/ml-1m --dataset ml-1m --n_neg_items 7

echo Preprocessing Amazon Beauty dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

echo Preprocessing Amazon Video Games dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Video_Games --n_neg_items 7

echo Preprocessing Amazon Electronics dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Electronics --n_neg_items 7

#echo Preprocessing Amazon Movies and TV dataset...
#python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Movies_and_TV --n_neg_items 7

echo Preprocessing completed!