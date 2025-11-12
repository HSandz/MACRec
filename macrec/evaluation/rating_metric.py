from math import sqrt
from macrec.evaluation.metric_shim import Accuracy as ShimAccuracy, MeanSquaredError as ShimMSE, MeanAbsoluteError as ShimMAE


class Accuracy(ShimAccuracy):
    """Thin wrapper around the shim Accuracy to keep API stable."""
    pass


class MSE(ShimMSE):
    """Mean squared error wrapper compatible with previous update/compute calls."""
    def update(self, output: dict) -> None:
        # delegate to shim: accepts output dict
        super().update(output=output)

    def compute(self):
        res = super().compute()
        return {'mse': res.get('mse', 0.0)}


class RMSE(ShimMSE):
    """Root mean squared error computed from MSE."""
    def update(self, output: dict) -> None:
        super().update(output=output)

    def compute(self):
        res = super().compute()
        mse = res.get('mse', 0.0)
        return {'rmse': sqrt(mse)}


class MAE(ShimMAE):
    def update(self, output: dict) -> None:
        super().update(output=output)

    def compute(self):
        res = super().compute()
        return {'mae': res.get('mae', 0.0)}
