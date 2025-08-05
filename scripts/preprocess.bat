@echo off
REM Windows batch file for preprocessing MACRec datasets

echo Preprocessing ml-100k dataset...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

echo Preprocessing Amazon Beauty dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

echo Preprocessing Amazon Software dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Software --n_neg_items 7

echo Creating sample dataset...
python main.py --main Sample --data_dir data/Beauty/test.csv --output_dir data/Beauty/test_1000.csv --random --samples 1000

echo Preprocessing completed!
pause
