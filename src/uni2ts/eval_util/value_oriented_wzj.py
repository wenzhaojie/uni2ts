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
# 评价指标定义（保持原有格式）
# ============================
@dataclass
class ValueOrientedNLLMetric(BaseMetricDefinition):
    """
    面向“关键事件”的加权 NLL 指标（含平滑正则），用于评估阶段。
    参数说明：
      - event_weight: 事件点的损失放大倍数（>1 更重视事件点）
      - threshold_ratio: 比例阈值法中的比例（与 event_detector='ratio_max' 搭配）
      - lambda_smooth: 平滑正则项权重（对预测均值相邻差分的 L1）
      - forecast_type: 兼容字段（保留，不强依赖），可忽略
      - event_detector: 'quantile' 或 'ratio_max'，事件判定策略（新增）
      - event_quantile: 分位数阈值法中的分位数（如 0.9，新增）
      - time_dim: 时间维度索引（默认 -1，新增）
      - eps: 数值稳定项（新增）
    """
    event_weight: float = 2.0
    threshold_ratio: float = 0.8
    lambda_smooth: float = 0.1
    forecast_type: str = "mean"

    # 新增可配置项（保持默认不改变你现有行为；但默认用 quantile 更鲁棒）
    event_detector: str = "quantile"   # 可选：'quantile' | 'ratio_max'
    event_quantile: float = 0.9
    time_dim: int = -1
    eps: float = 1e-12

    def __call__(self, axis: Optional[int] = 0) -> DirectMetric:
        """
        返回 GluonTS EV 框架期望的 DirectMetric。
        - name：指标名（带上关键超参，便于区分实验）
        - stat：单条样本统计函数（下方 value_oriented_nll_stat）
        - aggregate：对批次样本聚合（这里用均值）
        """
        return DirectMetric(
            name=(
                "ValueOrientedNLL"
                f"[event_weight={self.event_weight},smooth={self.lambda_smooth},"
                f"detector={self.event_detector},q={self.event_quantile}]"
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
            ),
            aggregate=Mean(axis=0),
        )


# ============================
# 指标的“单样本统计”函数
# ============================
def value_oriented_nll_stat(
    data,
    event_weight: float = 2.0,
    threshold_ratio: float = 0.8,
    lambda_smooth: float = 0.1,
    forecast_type: str = "mean",
    # 新增参数（与 Metric 保持一致）
    event_detector: str = "quantile",   # 'quantile' | 'ratio_max'
    event_quantile: float = 0.9,
    time_dim: int = -1,
    eps: float = 1e-12,
    **kwargs
):
    """
    单个样本的数值统计过程：
      1) 从 data 中解析 y_true（label/target）、forecast（或 mean/scale）
      2) 计算逐点 NLL：-log p(y | θ)
      3) 事件加权：基于 mask 感知的阈值（分位数或比例×最大值）
      4) 按有效位（prediction_mask ∩ observed_mask）归一化
      5) 平滑项：对预测均值在相邻有效位上的 L1 差分
      6) total = value_loss + lambda_smooth * smooth_loss
      7) 返回 np.array([total], dtype=np.float32)

    说明：
      - 评估阶段通常没有训练时的 PackedLoss 接口，这里做了与前文一致的“数值稳健化”处理。
      - 若 data 数据不全或分布无效，会返回 0.0，并打印 WARN，避免中断整个评估流程。
    """
    # -----------------------
    # 0) 工具函数（闭包内定义）
    # -----------------------
    def _to_tensor(x) -> torch.Tensor:
        """将 numpy 或 python 标量转为 torch.float32 张量。"""
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32)

    def _align_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        """将 mask 对齐/广播到与 ref 同形；返回 float32（便于乘法）。"""
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
        """合并预测/观测 mask：都在时取交集；只有一个则用那个；都无则返回 None。"""
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
        """mask 感知的最大值；无效位用 -inf（这里用 dtype 最小值）屏蔽。"""
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if mask is None:
            return x.max(dim=dim, keepdim=True).values
        very_small = torch.finfo(x.dtype).min
        x_masked = torch.where(mask > 0, x, torch.full_like(x, very_small))
        return x_masked.max(dim=dim, keepdim=True).values

    @torch.no_grad()
    def _masked_quantile(x: torch.Tensor, mask: Optional[torch.Tensor], q: float, dim: int) -> torch.Tensor:
        """
        mask 感知的分位数：无效位设为 NaN，使用 nanquantile 忽略。
        若全为 NaN，结果为 NaN；调用处需要兜底。
        """
        x = torch.where(torch.isfinite(x), x, torch.nan)
        if mask is not None:
            x = torch.where(mask > 0, x, torch.nan)
        return torch.nanquantile(x, q, dim=dim, keepdim=True)

    def _get_mean_from_distr(distr) -> Optional[torch.Tensor]:
        """尽量从分布对象获取“预测均值”（.mean 或 .loc）。"""
        m = getattr(distr, "mean", None)
        if isinstance(m, torch.Tensor):
            return m
        loc = getattr(distr, "loc", None)
        if isinstance(loc, torch.Tensor):
            return loc
        return None

    # -----------------------
    # 1) 取主字典（ChainMap 取 maps[0]）
    # -----------------------
    if isinstance(data, collections.ChainMap):
        main_data = data.maps[0]
    else:
        main_data = data

    # -----------------------
    # 2) 解析 y_true 与 forecast/distribution
    # -----------------------
    y_true = None
    if "label" in main_data:
        y_true = main_data["label"]
    elif "target" in main_data:
        y_true = main_data["target"]

    forecast = main_data.get("forecast", None)

    # 若没有 forecast，但有 mean(/scale)，构造一个“正态分布”占位分布
    # 说明：仅用于评估近似；scale 若缺失，取 1；若有则确保正值（>=1e-6）
    if forecast is None and ("mean" in main_data or "scale" in main_data):
        mean_arr = main_data.get("mean", None)
        scale_arr = main_data.get("scale", None)

        class _DummyNormal:
            def __init__(self, mean, scale):
                self._mean = _to_tensor(mean)
                self._scale = _to_tensor(scale)
                # 避免 scale 为 0 或负值
                self._scale = torch.clamp(self._scale, min=1e-6)

            def log_prob(self, x: torch.Tensor) -> torch.Tensor:
                x = _to_tensor(x).to(device=self._mean.device)
                # 正态分布对数密度：-0.5*((x-μ)/σ)^2 - log(σ) - 0.5*log(2π)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)

            @property
            def mean(self) -> torch.Tensor:
                return self._mean

        if mean_arr is None:
            print("[WARN] value_oriented_nll_stat: mean missing; cannot build dummy distribution, return 0.0")
            return np.zeros((1,), dtype=np.float32)

        if scale_arr is None:
            scale_arr = np.ones_like(mean_arr, dtype=np.float32)

        distr = _DummyNormal(mean_arr, scale_arr)
        forecast = type("DummyForecast", (), {"distribution": distr})()

    # -----------------------
    # 3) 基本合法性检查
    # -----------------------
    if y_true is None or forecast is None:
        print("[WARN] value_oriented_nll_stat: illegal data (missing target/forecast), return 0.0")
        return np.zeros((1,), dtype=np.float32)

    # 转 tensor
    y_true = _to_tensor(y_true)

    # 获取分布对象
    distr = None
    if hasattr(forecast, "distribution") and forecast.distribution is not None:
        distr = forecast.distribution
    elif hasattr(forecast, "mean") and hasattr(forecast, "scale"):
        # 兼容另一种 forecast 结构（同样用正态近似）
        class _DummyNormal2:
            def __init__(self, mean, scale):
                self._mean = _to_tensor(mean)
                self._scale = torch.clamp(_to_tensor(scale), min=1e-6)

            def log_prob(self, x):
                x = _to_tensor(x).to(device=self._mean.device)
                z = (x - self._mean) / self._scale
                return -0.5 * z.pow(2) - torch.log(self._scale) - 0.5 * np.log(2 * np.pi)

            @property
            def mean(self):
                return self._mean

        distr = _DummyNormal2(forecast.mean, forecast.scale)
    else:
        print("[DEBUG] forecast has no valid distribution/mean/scale")
        print("[WARN] value_oriented_nll_stat: illegal data (bad forecast), return 0.0")
        return np.zeros((1,), dtype=np.float32)

    try:
        # -----------------------
        # 4) 基础逐点 NLL
        # -----------------------
        log_prob = distr.log_prob(y_true)
        loss = -log_prob  # 形状与 y_true 一致
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        # -----------------------
        # 5) 取有效位（若存在）：prediction_mask ∩ observed_mask
        #    评估数据常见键名：prediction_mask / observed_mask / valid_mask（可选）
        # -----------------------
        pred_mask = main_data.get("prediction_mask", None)
        obs_mask = main_data.get("observed_mask", None)
        # 若上游已提供 valid_mask，也可使用；但我们仍优先用 ∩ 交集逻辑
        #（避免与已有 valid_mask 定义不一致，这里只做兜底）
        vm_from_upstream = main_data.get("valid_mask", None)

        # loss 为基准做对齐
        valid_mask = _merge_masks(pred_mask, obs_mask, loss)
        if valid_mask is None and vm_from_upstream is not None:
            valid_mask = _align_mask(_to_tensor(vm_from_upstream), loss)

        # -----------------------
        # 6) 事件阈值（mask 感知）
        # -----------------------
        if event_detector == "quantile":
            thr = _masked_quantile(y_true, valid_mask, event_quantile, dim=time_dim)
            # 若全无有效位 -> 分位数为 NaN，回退到比例阈值
            fallback = threshold_ratio * _masked_max(y_true, valid_mask, dim=time_dim)
            thr = torch.where(torch.isfinite(thr), thr, fallback)
        else:  # 'ratio_max'
            mx = _masked_max(y_true, valid_mask, dim=time_dim)
            thr = threshold_ratio * mx

        # -----------------------
        # 7) 事件加权并掩蔽无效位
        # -----------------------
        is_event = (y_true > thr).to(dtype=loss.dtype)
        weight = 1.0 + (event_weight - 1.0) * is_event
        weighted_loss = loss * weight

        if valid_mask is not None:
            weighted_loss = weighted_loss * valid_mask

        # -----------------------
        # 8) 按有效元素数归一化
        # -----------------------
        if valid_mask is None:
            valid_count = torch.tensor(weighted_loss.numel(), device=weighted_loss.device, dtype=weighted_loss.dtype)
        else:
            valid_count = valid_mask.sum()

        if (valid_count <= 0).item():
            # 无有效位：返回 0，避免打断评估流程
            print("[WARN] value_oriented_nll_stat: no valid elements, return 0.0")
            return np.zeros((1,), dtype=np.float32)

        value_loss = weighted_loss.sum() / (valid_count + eps)

        # -----------------------
        # 9) 平滑正则（对分布的预测均值）
        #    仅在相邻两时刻均有效的位置上计算 L1 差分
        # -----------------------
        smooth_loss = torch.tensor(0.0, device=weighted_loss.device, dtype=weighted_loss.dtype)
        mean_pred = _get_mean_from_distr(distr)
        if isinstance(mean_pred, torch.Tensor):
            mean_pred = torch.where(torch.isfinite(mean_pred), mean_pred, torch.zeros_like(mean_pred))

            # 构造相邻时刻索引（支持任意 time_dim）
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
                        smooth_loss = torch.tensor(0.0, device=weighted_loss.device, dtype=weighted_loss.dtype)
                else:
                    smooth_loss = torch.mean(torch.abs(diff))
            else:
                smooth_loss = torch.tensor(0.0, device=weighted_loss.device, dtype=weighted_loss.dtype)

        # -----------------------
        # 10) 总指标值并返回 numpy
        # -----------------------
        total_loss = value_loss + lambda_smooth * smooth_loss
        total_loss = torch.where(torch.isfinite(total_loss), total_loss, torch.zeros_like(total_loss))

        result = total_loss.item() if hasattr(total_loss, "item") else float(total_loss)
        if np.isnan(result):
            print("[WARN] value_oriented_nll_stat: got NaN total, return 0.0")
            return np.zeros((1,), dtype=np.float32)
        return np.array([result], dtype=np.float32)

    except Exception as e:
        # 任何异常，打印并返回 0，避免评估任务中断
        print(f"[ERROR] value_oriented_nll_stat: Exception {e}, return 0.0")
        return np.zeros((1,), dtype=np.float32)
