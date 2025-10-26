#! /bin/bash

echo Preprocessing ml-100k dataset...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

echo Preprocessing Amazon Digital Music dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Digital_Music --n_neg_items 7

echo Preprocessing Amazon Beauty dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

# echo Preprocessing Gowalla dataset...
# python main.py --main Preprocess --data_dir data/Gowalla --dataset gowalla --n_neg_items 7

# echo Preprocessing Yelp2018 dataset...
# python main.py --main Preprocess --data_dir data/Yelp2018 --dataset yelp2018 --n_neg_items 7

# echo Preprocessing Amazon Video Games dataset...
# python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Video_Games --n_neg_items 7

# echo Preprocessing Amazon Toys and Games dataset...
# python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Toys_and_Games --n_neg_items 7

# echo Preprocessing Amazon Movies and TV dataset...
# python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Movies_and_TV --n_neg_items 7

echo Preprocessing completed!