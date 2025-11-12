from abc import abstractmethod
from loguru import logger
import math


class RankMetric:
    """
    Base class for rank metrics without torchmetrics dependency.

    Subclasses should implement `metric_at_k(self, answer, label) -> dict`.
    """
    def __init__(self, topks: list[int] | int, *args, **kwargs):
        if isinstance(topks, int):
            topks = [topks]
        self.topks = topks
        # initialize counters
        for topk in self.topks:
            setattr(self, f'at{topk}', 0.0)
        self.total = 0

    def update(self, output: dict) -> None:
        answer = output['answer']
        label = output['label']

        # Handle string answers (from errors or fallback modes)
        if isinstance(answer, str):
            logger.warning(f"Received string answer instead of list: {answer}")
            metrics = {topk: 0 for topk in self.topks}
        else:
            metrics = self.metric_at_k(answer, label)

        for topk in self.topks:
            metric = metrics[topk]
            current = getattr(self, f'at{topk}')
            setattr(self, f'at{topk}', current + float(metric))

        self.total += 1

    def compute(self):
        result = {}
        for topk in self.topks:
            if self.total != 0:
                result[topk] = getattr(self, f'at{topk}') / float(self.total)
            else:
                result[topk] = 0
        return result

    @abstractmethod
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        """Calculate the rank metric at k.

        Args:
            `answer` (`list[int]`): The ranking given by the system.
            `label` (`int`): The ground truth answer.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `dict`: The rank metric at k.
        """
        raise NotImplementedError

class HitRatioAt(RankMetric):
    """
    Hit ratio at k. If the ground truth answer is in the top k, then the metric is 1, otherwise 0.
    """
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        # Convert label to Python int to handle numpy types
        label = int(label)
        for topk in self.topks:
            if label in answer[:topk]:
                result[topk] = 1
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'HR@{topk}': result[topk] for topk in self.topks}

class NDCGAt(RankMetric):
    """
    Normalized discounted cumulative gain at k. If the ground truth answer is in the top k, then the metric is 1 / log2(label position + 1), otherwise 0.
    """
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        # Convert label to Python int to handle numpy types
        label = int(label)
        for topk in self.topks:
            try:
                label_pos = answer.index(label) + 1
            except ValueError:
                label_pos = topk + 1
            if label_pos <= topk:
                # use math.log2 for a lightweight implementation
                result[topk] = 1.0 / math.log2(label_pos + 1.0)
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'NDCG@{topk}': result[topk] for topk in self.topks}

class MRRAt(RankMetric):
    """
    Mean reciprocal rank at k. If the ground truth answer is in the top k, then the metric is 1 / label position, otherwise 0.
    """
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        # Convert label to Python int to handle numpy types
        label = int(label)
        for topk in self.topks:
            try:
                label_pos = answer.index(label) + 1
            except ValueError:
                label_pos = topk + 1
            if label_pos <= topk:
                result[topk] = 1.0 / float(label_pos)
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'MRR@{topk}': result[topk] for topk in self.topks}
