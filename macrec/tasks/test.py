import numpy as np
import pandas as pd
from argparse import ArgumentParser

from macrec.tasks.evaluate import EvaluateTask
from macrec.utils import init_all_seeds

class TestTask(EvaluateTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = EvaluateTask.parse_task_args(parser)
        parser.add_argument('--random', action='store_true', help='Whether to randomly sample test data')
        parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
        parser.add_argument('--offset', type=int, default=0, help='Offset of samples, only works when random is False')
        parser.add_argument('--last', action='store_true', help='Sample from the end of the dataset instead of the beginning')
        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = super().get_data(data_file, max_his)
        
        if self.random:
            sample_idx = np.random.choice(len(df), min(self.samples, len(df)), replace=False)
            df = df.iloc[sample_idx].reset_index(drop=True)
        else:
            if self.last:
                start_idx = max(0, len(df) - self.samples)
                df = df.iloc[start_idx: start_idx + self.samples].reset_index(drop=True)
            else:
                df = df.iloc[self.offset: self.offset + self.samples].reset_index(drop=True)
        
        return df

    def run(self, random: bool, samples: int, offset: int, last: bool = False, *args, **kwargs):
        self.sampled = True
        self.random = random
        if self.random:
            init_all_seeds(2024)
        self.samples = samples
        self.offset = offset
        self.last = last
        super().run(*args, **kwargs)

if __name__ == '__main__':
    TestTask().launch()
