# uni2ts/eval_util/value_oriented_wzj.py

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric

@dataclass
class ValueOrientedNLLMetric(BaseMetricDefinition):
    """
    Value-oriented weighted NLL loss for evaluation,
    with event region emphasis and smoothness penalty.
    """
    event_weight: float = 2.0
    threshold_ratio: float = 0.8
    lambda_smooth: float = 0.1
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"ValueOrientedNLL[event_weight={self.event_weight},smooth={self.lambda_smooth}]",
            stat=partial(
                value_oriented_nll_stat,
                event_weight=self.event_weight,
                threshold_ratio=self.threshold_ratio,
                lambda_smooth=self.lambda_smooth,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )

def value_oriented_nll_stat(
    y_true,
    forecast,
    event_weight: float = 2.0,
    threshold_ratio: float = 0.8,
    lambda_smooth: float = 0.1,
    forecast_type: str = "mean",
    **kwargs
):
    """
    GluonTS evaluation-style metric: y_true, forecast (distribution object).
    """
    # 支持 torch/numpy 输入
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if hasattr(forecast, "distribution") and forecast.distribution is not None:
        distr = forecast.distribution
    elif hasattr(forecast, "mean") and hasattr(forecast, "scale"):  # fallback
        class DummyDistr:
            def log_prob(self, x):
                mean = torch.from_numpy(forecast.mean)
                scale = torch.from_numpy(forecast.scale)
                return -0.5 * ((x - mean) / scale) ** 2 - scale.log() - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self):
                return torch.from_numpy(forecast.mean)
        distr = DummyDistr()
    else:
        raise ValueError("Unsupported forecast type for ValueOrientedNLLMetric.")

    # NLL loss
    log_prob = distr.log_prob(y_true)
    loss = -log_prob

    # Event weighting
    threshold = threshold_ratio * y_true.max(dim=-1, keepdim=True)[0]
    is_event = (y_true > threshold).float()
    weight = 1.0 + (event_weight - 1.0) * is_event
    weighted_loss = loss * weight

    # Smoothness penalty (on mean)
    if hasattr(distr, "mean"):
        mean_pred = distr.mean
        diff = mean_pred[..., 1:] - mean_pred[..., :-1]
        smooth_loss = torch.mean(torch.abs(diff))
    else:
        smooth_loss = 0.0

    # Reduction
    total_loss = weighted_loss.mean() + lambda_smooth * smooth_loss
    return total_loss.item() if hasattr(total_loss, "item") else float(total_loss)
