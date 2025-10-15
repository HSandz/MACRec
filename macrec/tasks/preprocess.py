from argparse import ArgumentParser
import os

from macrec.tasks.base import Task
from macrec.dataset import ml100k_process_data, amazon_process_data, ml100k_process_data_with_retrieval, amazon_process_data_with_retrieval
from macrec.dataset.ml100k import process_data
from macrec.utils import init_all_seeds

class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='input file')
        parser.add_argument('--dataset', type=str, required=True, choices=['ml-100k', 'amazon'], help='dataset type')
        parser.add_argument('--amazon_category', type=str, help='Amazon category (e.g., Beauty, Books, Movies_and_TV, etc.). Required when dataset is amazon.')
        parser.add_argument('--n_neg_items', type=int, default=7, help='numbers of negative items')
        parser.add_argument('--sampling_strategy', type=str, default='negative', choices=['negative', 'retrieval'], help='Sampling strategy: negative sampling or retrieval-based')
        parser.add_argument('--n_retrieval_items', type=int, default=20, help='Number of retrieval items when using retrieval-based sampling')
        return parser

    def run(self, data_dir: str, dataset: str, amazon_category: str = None, n_neg_items: int = 7, sampling_strategy: str = 'negative', n_retrieval_items: int = 20):
        init_all_seeds(2024)
        if dataset == 'ml-100k':
            process_data(data_dir, n_neg_items=n_neg_items, sampling_strategy=sampling_strategy, n_retrieval_items=n_retrieval_items)
        elif dataset == 'amazon':
            if amazon_category is None:
                raise ValueError("--amazon_category is required when dataset is 'amazon'. Please specify a category like 'Beauty', 'Books', etc.")
            
            # Construct the data directory path with the specified category
            category_data_dir = os.path.join(data_dir, amazon_category)
            if sampling_strategy == 'retrieval':
                amazon_process_data_with_retrieval(category_data_dir, n_retrieval_items)
            else:
                amazon_process_data(category_data_dir, n_neg_items)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    PreprocessTask().launch()
