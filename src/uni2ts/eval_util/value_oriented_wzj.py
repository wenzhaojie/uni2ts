# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import partial
from typing import Optional
import collections

import numpy as np
import torch
from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric


# ============================
# Metric 定义（参照 MedianMSE 风格）
# ============================
@dataclass
class ValueOrientedNLLMetric(BaseMetricDefinition):
    event_weight: float = 2.0
    threshold_ratio: float = 0.8
    lambda_smooth: float = 0.1
    forecast_type: str = "mean"

    # 事件判定与数值配置
    event_detector: str = "quantile"   # 'quantile' | 'ratio_max'
    event_quantile: float = 0.9
    time_dim: int = -1
    eps: float = 1e-12

    # 调试选项：无预测时用 y_true 作为 mean、scale=1 作兜底
    allow_identity_fallback: bool = False

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=(
                "ValueOrientedNLL"
                f"[w={self.event_weight},s={self.lambda_smooth},"
                f"det={self.event_detector},q={self.event_quantile}]"
            ),
            stat=partial(
                value_oriented_nll_stat,
                event_weight=self.event_weight,
                threshold_ratio=self.threshold_ratio,
                lambda_smooth=self.lambda_smooth,
                forecast_type=self.forecast_type,
                event_detector=self.event_detector,
                event_quantile=self.event_quantile,
                time_dim=self.time_dim,
                eps=self.eps,
                allow_identity_fallback=self.allow_identity_fallback,
            ),
            aggregate=Mean(axis=axis),
        )


# ============================
# 单样本统计函数（健壮版）
# ============================
def value_oriented_nll_stat(
    data,
    event_weight: float = 2.0,
    threshold_ratio: float = 0.8,
    lambda_smooth: float = 0.1,
    forecast_type: str = "mean",
    event_detector: str = "quantile",
    event_quantile: float = 0.9,
    time_dim: int = -1,
    eps: float = 1e-12,
    allow_identity_fallback: bool = False,
    **kwargs
):
    """Compute value-oriented NLL with event weighting and smoothness, mask-aware."""
    # ---- helpers ----
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32)

    def _align_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = _to_tensor(mask).to(device=ref.device)
        while mask.dim() < ref.dim():
            mask = mask.unsqueeze(-1)
        while mask.dim() > ref.dim():
            mask = mask.squeeze(-1)
        if mask.shape != ref.shape:
            mask = mask.expand_as(ref)
        return mask

    def _merge_masks(pm: Optional[torch.Tensor], om: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        pm = _align_mask(pm, ref)
        om = _align_mask(om, ref)
        if pm is not None and om is not None:
            return ((pm > 0) & (om > 0)).to(dtype=ref.dtype)
        if pm is not None:
            return (pm > 0).to(dtype=ref.dtype)
        if om is not None:
            return (om > 0).to(dtype=ref.dtype)
        return None

    @torch.no_grad()
    def _masked_max(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if mask is None:
            return x.max(dim=dim, keepdim=True).values
        very_small = torch.finfo(x.dtype).min
        x_masked = torch.where(mask > 0, x, torch.full_like(x, very_small))
        return x_masked.max(dim=dim, keepdim=True).values

    @torch.no_grad()
    def _masked_quantile(x: torch.Tensor, mask: Optional[torch.Tensor], q: float, dim: int) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, torch.nan)
        if mask is not None:
            x = torch.where(mask > 0, x, torch.nan)
        return torch.nanquantile(x, q, dim=dim, keepdim=True)

    def _get_mean_from_distr(distr) -> Optional[torch.Tensor]:
        m = getattr(distr, "mean", None)
        if isinstance(m, torch.Tensor):
            return m
        loc = getattr(distr, "loc", None)
        if isinstance(loc, torch.Tensor):
            return loc
        return None

    # ---- 1) main dict ----
    if isinstance(data, collections.ChainMap):
        main_data = data.maps[0]
    else:
        main_data = data

    # ---- 2) y_true (support MaskedArray) ----
    y_true_raw = None
    for k in ("label", "target", "y", "future_target", "gt"):
        if k in main_data:
            y_true_raw = main_data[k]
            break
    if y_true_raw is None:
        print("[WARN] value_oriented_nll_stat: missing target-like keys, return 0.0")
        return np.zeros((1,), dtype=np.float32)

    observed_mask_from_label = None
    if isinstance(y_true_raw, np.ma.MaskedArray):
        observed_mask_from_label = torch.as_tensor(~y_true_raw.mask, dtype=torch.float32)
        y_true = torch.as_tensor(y_true_raw.filled(np.nan), dtype=torch.float32)
    else:
        y_true = _to_tensor(y_true_raw)

    # ---- 3) distribution from forecast ----
    distr = None
    forecast = main_data.get("forecast", None)

    if forecast is not None and hasattr(forecast, "distribution") and forecast.distribution is not None:
        distr = forecast.distribution

    elif forecast is not None and hasattr(forecast, "mean") and hasattr(forecast, "scale"):
        class _DummyNormal2:
            def __init__(self, mean, scale):
                self._mean = _to_tensor(mean)
                self._scale = torch.clamp(_to_tensor(scale), min=1e-6)
            def log_prob(self, x):
                x = _to_tensor(x).to(self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self): return self._mean
        distr = _DummyNormal2(forecast.mean, forecast.scale)

    elif forecast is not None and hasattr(forecast, "samples"):
        samples = _to_tensor(forecast.samples)
        mu = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False).clamp_min(1e-6)
        class _DummyNormal3:
            def __init__(self, mean, scale): self._mean, self._scale = mean, scale
            def log_prob(self, x):
                x = _to_tensor(x).to(self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self): return self._mean
        distr = _DummyNormal3(mu, std)

    elif isinstance(forecast, (np.ndarray, torch.Tensor)):
        mean_arr = forecast
        scale_arr = main_data.get("scale", None) or main_data.get("std", None)
        if scale_arr is None:
            scale_arr = np.ones_like(mean_arr, dtype=np.float32) if isinstance(mean_arr, np.ndarray) \
                        else torch.ones_like(_to_tensor(mean_arr))
        class _DummyNormal4:
            def __init__(self, mean, scale):
                self._mean = _to_tensor(mean)
                self._scale = torch.clamp(_to_tensor(scale), min=1e-6)
            def log_prob(self, x):
                x = _to_tensor(x).to(self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self): return self._mean
        distr = _DummyNormal4(mean_arr, scale_arr)

    elif any(k in main_data for k in ("y_pred", "prediction", "yhat", "point_forecast")):
        mean_arr = main_data.get("y_pred", None) or main_data.get("prediction", None) \
                   or main_data.get("yhat", None) or main_data.get("point_forecast", None)
        scale_arr = main_data.get("scale", None) or main_data.get("std", None)
        if scale_arr is None:
            scale_arr = np.ones_like(mean_arr, dtype=np.float32) if isinstance(mean_arr, np.ndarray) \
                        else torch.ones_like(_to_tensor(mean_arr))
        class _DummyNormal5:
            def __init__(self, mean, scale):
                self._mean = _to_tensor(mean)
                self._scale = torch.clamp(_to_tensor(scale), min=1e-6)
            def log_prob(self, x):
                x = _to_tensor(x).to(self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self): return self._mean
        distr = _DummyNormal5(mean_arr, scale_arr)

    elif ("mean" in main_data) or ("scale" in main_data) or ("std" in main_data):
        mean_arr = main_data.get("mean", None)
        scale_arr = main_data.get("scale", None) or main_data.get("std", None)
        if mean_arr is not None:
            if scale_arr is None:
                scale_arr = np.ones_like(mean_arr, dtype=np.float32) if isinstance(mean_arr, np.ndarray) \
                            else torch.ones_like(_to_tensor(mean_arr))
            class _DummyNormal6:
                def __init__(self, mean, scale):
                    self._mean = _to_tensor(mean)
                    self._scale = torch.clamp(_to_tensor(scale), min=1e-6)
                def log_prob(self, x):
                    x = _to_tensor(x).to(self._mean.device)
                    z = (x - self._mean) / self._scale
                    return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
                @property
                def mean(self): return self._mean
            distr = _DummyNormal6(mean_arr, scale_arr)

    if distr is None and allow_identity_fallback:
        class _IdentityNormal:
            def __init__(self, y):
                self._mean = _to_tensor(y)
                self._scale = torch.ones_like(self._mean)
            def log_prob(self, x):
                x = _to_tensor(x).to(self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)
            @property
            def mean(self): return self._mean
        distr = _IdentityNormal(y_true)
        print("[INFO] value_oriented_nll_stat: using identity fallback (mean=y_true, scale=1)")

    if distr is None:
        print("[WARN] value_oriented_nll_stat: missing forecast/mean/scale/samples, return 0.0")
        return np.zeros((1,), dtype=np.float32)

    # ---- 4) NLL ----
    try:
        log_prob = distr.log_prob(y_true)
        loss = -log_prob
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        # ---- 5) valid mask ----
        pred_mask = main_data.get("prediction_mask", None)
        obs_mask  = main_data.get("observed_mask", None)
        vm_from_upstream = main_data.get("valid_mask", None)
        if obs_mask is None and 'observed_mask_from_label' in locals() and observed_mask_from_label is not None:
            obs_mask = observed_mask_from_label

        valid_mask = _merge_masks(pred_mask, obs_mask, loss)
        if valid_mask is None and vm_from_upstream is not None:
            valid_mask = _align_mask(_to_tensor(vm_from_upstream), loss)

        # ---- 6) threshold ----
        if event_detector == "quantile":
            thr = _masked_quantile(y_true, valid_mask, event_quantile, dim=time_dim)
            fallback = threshold_ratio * _masked_max(y_true, valid_mask, dim=time_dim)
            thr = torch.where(torch.isfinite(thr), thr, fallback)
        else:
            mx = _masked_max(y_true, valid_mask, dim=time_dim)
            thr = threshold_ratio * mx

        # ---- 7) weight + mask ----
        is_event = (y_true > thr).to(dtype=loss.dtype)
        weight = 1.0 + (event_weight - 1.0) * is_event
        weighted_loss = loss * weight
        if valid_mask is not None:
            weighted_loss = weighted_loss * valid_mask

        # ---- 8) normalize ----
        if valid_mask is None:
            valid_count = torch.tensor(weighted_loss.numel(), device=weighted_loss.device, dtype=weighted_loss.dtype)
        else:
            valid_count = valid_mask.sum()
        if (valid_count <= 0).item():
            print("[WARN] value_oriented_nll_stat: no valid elements, return 0.0")
            return np.zeros((1,), dtype=np.float32)

        value_loss = weighted_loss.sum() / (valid_count + eps)

        # ---- 9) smoothness ----
        smooth_loss = torch.tensor(0.0, device=weighted_loss.device, dtype=weighted_loss.dtype)
        mean_pred = _get_mean_from_distr(distr)
        if isinstance(mean_pred, torch.Tensor):
            mean_pred = torch.where(torch.isfinite(mean_pred), mean_pred, torch.zeros_like(mean_pred))
            T = mean_pred.size(time_dim)
            if T >= 2:
                idx_t = torch.arange(1, T, device=mean_pred.device)
                idx_tm1 = torch.arange(0, T - 1, device=mean_pred.device)
                diff = mean_pred.index_select(time_dim, idx_t) - mean_pred.index_select(time_dim, idx_tm1)
                if valid_mask is not None:
                    vm_t = valid_mask.index_select(time_dim, idx_t)
                    vm_tm1 = valid_mask.index_select(time_dim, idx_tm1)
                    pair_mask = (vm_t > 0) & (vm_tm1 > 0)
                    pair_count = pair_mask.sum()
                    if (pair_count > 0).item():
                        smooth_loss = (torch.abs(diff) * pair_mask.to(diff.dtype)).sum() / (pair_count + eps)
                else:
                    smooth_loss = torch.mean(torch.abs(diff))

        total_loss = value_loss + lambda_smooth * smooth_loss
        total_loss = torch.where(torch.isfinite(total_loss), total_loss, torch.zeros_like(total_loss))

        result = total_loss.item() if hasattr(total_loss, "item") else float(total_loss)
        if np.isnan(result):
            print("[WARN] value_oriented_nll_stat: got NaN total, return 0.0")
            return np.zeros((1,), dtype=np.float32)
        return np.array([result], dtype=np.float32)

    except Exception as e:
        print(f"[ERROR] value_oriented_nll_stat: Exception {e}, return 0.0")
        return np.zeros((1,), dtype=np.float32)
