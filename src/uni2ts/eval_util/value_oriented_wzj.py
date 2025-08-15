from dataclasses import dataclass
from functools import partial
from typing import Optional, Mapping

import numpy as np

from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric


# ---------------------------
# 逐点统计：价值导向 NLL（与 Packed 版思路一致）
# 期望 data 至少包含:
#   - "target": (..., pred_len, dim) 或 (..., pred_len)  [时间维为倒数第二维]
# 可选:
#   - "log_prob": 与 target 同形状的逐点 log p(y)
#   - "nll": 与 target 同形状的逐点 NLL
#   - "mean": 逐点均值（用于平滑正则以及必要时的高斯近似）
#   - "variance" 或 "scale": 高斯近似所需
#   - "prediction_mask": (..., pred_len)  1=评估区
#   - "observed_mask": 与 target 同形状 1=有观测
# 备注：本函数按单条 time series 被反复调用（DirectMetric.update 单样本调用）
# ---------------------------
def _value_oriented_nll_stat(
    data: Mapping[str, np.ndarray],
    *,
    spike_quantile: float,
    spike_weight: float,
    smooth_lambda: float,
    apply_only_on_pred_mask: bool,
    eps: float,
) -> np.ndarray:
    # 取 target，并标准化为至少 2D（time, dim）
    target = np.asarray(data["target"])
    if target.ndim == 1:
        target = target[:, None]  # -> (T, 1)
    # 约定：时间维为倒数第二维
    T, D = target.shape[-2], target.shape[-1]

    # 掩码准备
    pred_mask = np.asarray(data.get("prediction_mask", np.ones((T,), dtype=bool)))
    if pred_mask.ndim == 0:
        pred_mask = np.full((T,), bool(pred_mask))
    obs_mask = np.asarray(data.get("observed_mask", np.ones_like(target, dtype=bool)))
    if obs_mask.shape != target.shape:
        # 允许提供 (T,) 或 (T,1) 的观测掩码，自动广播到 (T,D)
        if obs_mask.ndim == 1 and obs_mask.shape[0] == T:
            obs_mask = np.repeat(obs_mask[:, None], D, axis=1)
        elif obs_mask.ndim == 2 and obs_mask.shape == (T, 1):
            obs_mask = np.repeat(obs_mask, D, axis=1)
        else:
            raise ValueError(f"observed_mask shape {obs_mask.shape} incompatible with target {target.shape}")

    # 1) 基础 NLL：优先用 log_prob / nll；否则用高斯近似
    base_nll = None
    if "log_prob" in data:
        logp = np.asarray(data["log_prob"])
        if logp.ndim == 1:
            logp = logp[:, None]
        base_nll = -logp
    elif "nll" in data:
        nll = np.asarray(data["nll"])
        if nll.ndim == 1:
            nll = nll[:, None]
        base_nll = nll
    else:
        # Gaussian 近似： 0.5*log(2πσ^2) + (y-μ)^2/(2σ^2)
        if "mean" not in data:
            raise KeyError("Need one of {'log_prob','nll'} or {'mean' and ('variance' or 'scale')} in data.")
        mean = np.asarray(data["mean"])
        if mean.ndim == 1:
            mean = mean[:, None]
        if mean.shape != target.shape:
            if mean.shape == (T, 1):
                mean = np.repeat(mean, D, axis=1)
            else:
                raise ValueError(f"mean shape {mean.shape} incompatible with target {target.shape}")

        if "variance" in data:
            var = np.asarray(data["variance"])
            if var.ndim == 1:
                var = var[:, None]
            if var.shape != target.shape:
                if var.shape == (T, 1):
                    var = np.repeat(var, D, axis=1)
                else:
                    raise ValueError(f"variance shape {var.shape} incompatible with target {target.shape}")
            sigma2 = np.clip(var, eps, None)
        elif "scale" in data:
            scale = np.asarray(data["scale"])
            if scale.ndim == 1:
                scale = scale[:, None]
            if scale.shape != target.shape:
                if scale.shape == (T, 1):
                    scale = np.repeat(scale, D, axis=1)
                else:
                    raise ValueError(f"scale shape {scale.shape} incompatible with target {target.shape}")
            sigma2 = np.clip(scale ** 2, eps, None)
        else:
            raise KeyError("Need 'variance' or 'scale' along with 'mean' for Gaussian NLL fallback.")

        base_nll = 0.5 * (np.log(2.0 * np.pi * sigma2) + ((target - mean) ** 2) / sigma2)

    # 数值清理
    base_nll = np.nan_to_num(base_nll, nan=0.0, posinf=1e6, neginf=0.0)

    # 2) 突增识别：|Δy_t|，阈值取分位数（默认仅在预测区且成对可观测）
    # 构造逐维可观测与预测掩码
    pred_mask_exp = np.repeat(pred_mask[:, None], D, axis=1)
    obs_t = obs_mask.astype(bool)
    obs_tm1 = np.zeros_like(obs_t, dtype=bool)
    obs_tm1[1:, :] = obs_t[:-1, :]

    pair_mask = obs_t & obs_tm1
    if apply_only_on_pred_mask:
        pm_tm1 = np.zeros_like(pred_mask_exp, dtype=bool)
        pm_tm1[1:, :] = pred_mask_exp[:-1, :]
        pair_mask &= pred_mask_exp & pm_tm1

    delta = np.zeros_like(target, dtype=float)
    delta[1:, :] = np.abs(target[1:, :] - target[:-1, :])

    valid_delta = delta[pair_mask]
    if valid_delta.size > 0:
        thresh = np.quantile(valid_delta, spike_quantile)
    else:
        thresh = np.inf

    spike_mask = (delta > thresh) & pair_mask
    spike_w = 1.0 + (spike_weight - 1.0) * spike_mask.astype(float)

    # 3) 平滑正则：对 mean 的 (μ_t - μ_{t-1})^2；无 mean 则为 0
    reg_per_token = np.zeros_like(base_nll)
    if smooth_lambda > 0.0:
        mean = None
        if "mean" in data:
            mean = np.asarray(data["mean"])
            if mean.ndim == 1:
                mean = mean[:, None]
            if mean.shape != target.shape:
                if mean.shape == (T, 1):
                    mean = np.repeat(mean, D, axis=1)
                else:
                    raise ValueError(f"mean shape {mean.shape} incompatible with target {target.shape}")

        if mean is not None:
            dmu = np.zeros_like(mean, dtype=float)
            dmu[1:, :] = mean[1:, :] - mean[:-1, :]

            pm_tm1 = np.zeros_like(pred_mask_exp, dtype=bool)
            pm_tm1[1:, :] = pred_mask_exp[:-1, :]
            pair_mask_reg = pred_mask_exp & pm_tm1 & obs_t & obs_tm1

            reg_pair = (dmu ** 2) * pair_mask_reg.astype(float)

            # 分摊回相邻两个时刻
            reg_per_token[1:, :] += 0.5 * smooth_lambda * reg_pair[1:, :]
            reg_per_token[:-1, :] += 0.5 * smooth_lambda * reg_pair[1:, :]

    # 4) 合成逐点度量；仅在预测∧观测上有效，其他位置置 0，交给聚合器处理
    point_metric = base_nll * spike_w + reg_per_token
    eff_mask = (pred_mask_exp & obs_mask).astype(float)
    return point_metric * eff_mask  # 逐点返回，Mean(axis=...) 再做聚合


@dataclass
class ValueOrientedNLLMetric(BaseMetricDefinition):
    """
    价值导向 NLL 评估指标（DirectMetric 版本）：
      - 对 |Δtarget| 超过分位数阈值的“突增点”放大 NLL
      - 加入对预测均值的平滑正则
    兼容多种输入字段：log_prob / nll / (mean + variance|scale)
    """

    spike_quantile: float = 0.9
    spike_weight: float = 3.0
    smooth_lambda: float = 0.0
    apply_only_on_pred_mask: bool = True
    eps: float = 1e-8

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="ValueOrientedNLL",
            stat=partial(
                _value_oriented_nll_stat,
                spike_quantile=self.spike_quantile,
                spike_weight=self.spike_weight,
                smooth_lambda=self.smooth_lambda,
                apply_only_on_pred_mask=self.apply_only_on_pred_mask,
                eps=self.eps,
            ),
            aggregate=Mean(axis=axis),
        )
