from argparse import ArgumentParser
import os

from macrec.tasks.base import Task
from macrec.dataset import ml100k_process_data, amazon_process_data, ml1m_process_data, gowalla_process_data, yelp2018_process_data
from macrec.utils import init_all_seeds

class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='input file')
        parser.add_argument('--dataset', type=str, required=True, choices=['ml-100k', 'ml-1m', 'amazon', 'gowalla', 'yelp2018'], help='dataset type')
        parser.add_argument('--amazon_category', type=str, help='Amazon category (e.g., Beauty, Books, Movies_and_TV, etc.). Required when dataset is amazon.')
        parser.add_argument('--n_neg_items', type=int, default=7, help='numbers of negative items')
        return parser

    def run(self, data_dir: str, dataset: str, amazon_category: str = None, n_neg_items: int = 7):
        init_all_seeds(2024)
        if dataset == 'ml-100k':
            ml100k_process_data(data_dir, n_neg_items)
        elif dataset == 'ml-1m':
            ml1m_process_data(data_dir, n_neg_items)
        elif dataset == 'amazon':
            if amazon_category is None:
                raise ValueError("--amazon_category is required when dataset is 'amazon'. Please specify a category like 'Beauty', 'Books', etc.")
            
            # Construct the data directory path with the specified category
            category_data_dir = os.path.join(data_dir, amazon_category)
            amazon_process_data(category_data_dir, n_neg_items)
        elif dataset == 'gowalla':
            gowalla_process_data(data_dir, n_neg_items)
        elif dataset == 'yelp2018':
            yelp2018_process_data(data_dir, n_neg_items)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    PreprocessTask().launch()
