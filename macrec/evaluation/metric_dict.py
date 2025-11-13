from loguru import logger
from macrec.evaluation.metric_shim import Metric


class MetricDict:
    """A lightweight container for metrics compatible with the project's usage.

    This mirrors the previous behaviour expected by the codebase but avoids
    depending on torchmetrics so we don't pull in heavy submodules like
    text/bert which require `transformers`.
    """
    def __init__(self, metrics: dict[str, Metric] = {}):
        self.metrics: dict[str, Metric] = metrics

    def add(self, name: str, metric: Metric):
        self.metrics[name] = metric

    def update(self, output: dict, prefix: str = '') -> str:
        updated_metrics = []
        for metric_name, metric in self.metrics.items():
            if not metric_name.startswith(prefix):
                continue
            metric.update(output)
            computed = metric.compute()
            if len(computed) == 1:
                computed_val = next(iter(computed.values()))
                logger.debug(f'{metric_name}: {computed_val:.4f}')
            else:
                logger.debug(f'{metric_name}:')
                for key, value in computed.items():
                    logger.debug(f'{key}: {value:.4f}')

        # return the first metric with prefix (same behaviour as before)
        for metric_name, metric in self.metrics.items():
            if not metric_name.startswith(prefix):
                continue
            computed = metric.compute()
            computed_val = next(iter(computed.values()))
            return f'{metric_name}: {computed_val:.4f}'

        return ''

    def compute(self):
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric.compute()
        return result

    def report(self):
        result = self.compute()
        for metric_name, metric in result.items():
            if len(metric) == 1:
                metric_val = next(iter(metric.values()))
                logger.success(f'{metric_name}: {metric_val:.4f}')
            else:
                logger.success(f'{metric_name}:')
                for key, value in metric.items():
                    logger.success(f'{key}: {value:.4f}')
