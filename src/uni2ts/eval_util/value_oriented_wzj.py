from dataclasses import dataclass
from functools import partial
from typing import Optional
import collections

import numpy as np
import torch
from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric

@dataclass
class ValueOrientedNLLMetric(BaseMetricDefinition):
    event_weight: float = 2.0
    threshold_ratio: float = 0.8
    lambda_smooth: float = 0.1
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = 0) -> DirectMetric:
        return DirectMetric(
            name=f"ValueOrientedNLL[event_weight={self.event_weight},smooth={self.lambda_smooth}]",
            stat=partial(
                value_oriented_nll_stat,
                event_weight=self.event_weight,
                threshold_ratio=self.threshold_ratio,
                lambda_smooth=self.lambda_smooth,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=0),
        )

def value_oriented_nll_stat(
    data,
    event_weight: float = 2.0,
    threshold_ratio: float = 0.8,
    lambda_smooth: float = 0.1,
    forecast_type: str = "mean",
    **kwargs
):
    import collections
    # 只用主 dict
    if isinstance(data, collections.ChainMap):
        main_data = data.maps[0]
    else:
        main_data = data

    y_true = None
    forecast = None

    if "label" in main_data:
        y_true = main_data["label"]
    elif "target" in main_data:
        y_true = main_data["target"]

    if "forecast" in main_data:
        forecast = main_data["forecast"]
    elif "mean" in main_data:
        # 构造 dummy forecast
        class DummyDistr:
            def log_prob(self, x):
                mean = torch.from_numpy(main_data["mean"]) if isinstance(main_data["mean"], np.ndarray) else torch.as_tensor(main_data["mean"])
                scale = torch.ones_like(mean)
                return -0.5 * ((x - mean) / scale) ** 2 - scale.log() - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self):
                return torch.from_numpy(main_data["mean"]) if isinstance(main_data["mean"], np.ndarray) else torch.as_tensor(main_data["mean"])
        forecast = type("DummyForecast", (), {"distribution": DummyDistr()})()
    else:
        forecast = None

    if y_true is None or forecast is None:
        print("[WARN] value_oriented_nll_stat: illegal data, return 0.0")
        return np.zeros((1,), dtype=np.float32)

    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    distr = None
    if hasattr(forecast, "distribution") and forecast.distribution is not None:
        distr = forecast.distribution
    elif hasattr(forecast, "mean") and hasattr(forecast, "scale"):
        class DummyDistr2:
            def log_prob(self, x):
                mean = torch.from_numpy(forecast.mean)
                scale = torch.from_numpy(forecast.scale)
                return -0.5 * ((x - mean) / scale) ** 2 - scale.log() - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self):
                return torch.from_numpy(forecast.mean)
        distr = DummyDistr2()
    else:
        print("[DEBUG] forecast has no valid distribution/mean/scale")
        print("[WARN] value_oriented_nll_stat: illegal data, return 0.0")
        return np.zeros((1,), dtype=np.float32)

    try:
        log_prob = distr.log_prob(y_true)
        loss = -log_prob

        threshold = threshold_ratio * y_true.max(dim=-1, keepdim=True)[0]
        is_event = (y_true > threshold).float()
        weight = 1.0 + (event_weight - 1.0) * is_event
        weighted_loss = loss * weight

        if hasattr(distr, "mean"):
            mean_pred = distr.mean
            diff = mean_pred[..., 1:] - mean_pred[..., :-1]
            smooth_loss = torch.mean(torch.abs(diff))
        else:
            smooth_loss = 0.0

        total_loss = weighted_loss.mean() + lambda_smooth * smooth_loss
        result = total_loss.item() if hasattr(total_loss, "item") else float(total_loss)
        if np.isnan(result):
            print("[WARN] value_oriented_nll_stat: got nan, return 0.0")
            return np.zeros((1,), dtype=np.float32)
        return np.array([result], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] value_oriented_nll_stat: Exception {e}, return 0.0")
        return np.zeros((1,), dtype=np.float32)
