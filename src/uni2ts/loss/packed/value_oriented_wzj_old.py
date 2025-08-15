# -*- coding: utf-8 -*-
from typing import Optional
import torch
from uni2ts.loss.packed import PackedDistributionLoss


class PackedValueOrientedNLLLoss(PackedDistributionLoss):
    """
    面向“关键事件”的分布式时间序列 NLL 损失（Value-oriented NLL）。
    由三部分组成：
      1) 基础 NLL：-log p(y | θ)
      2) 事件区间加权：对超过阈值的高价值点放大权重（可选分位数阈值/最大值比例两种策略）
      3) 平滑正则：对模型预测均值(或loc)的相邻时间步差分做 L1，鼓励输出更平滑

    关键特性（较原始实现的改进处）：
      - 事件阈值/平滑项均为“mask 感知”（仅在有效位置上计算）
      - 有效样本数使用 预测区∩观测区 的交集
      - 对 pred.log_prob 与 target/mean 的 NaN/Inf 做数值防护，避免训练中断
      - 阈值策略可切换：'quantile'（默认，鲁棒）或 'ratio_max'（比例×最大值）
    """

    def __init__(
        self,
        lambda_smooth: float = 0.1,         # 平滑正则项的权重 λ
        event_weight: float = 2.0,          # 事件点的损失放大倍数（>1 表示更重视事件点）
        threshold_ratio: float = 0.8,       # ratio_max 模式下的阈值比例（thr = ratio * max）
        event_detector: str = "quantile",   # 'quantile' | 'ratio_max'，事件检测策略
        event_quantile: float = 0.9,        # quantile 模式下的分位数（如 0.9）
        time_dim: int = -1,                 # 时间维度索引（一般是 -1）
        eps: float = 1e-12,                 # 数值稳定用的极小值
    ):
        super().__init__()
        assert event_detector in ("quantile", "ratio_max")
        self.lambda_smooth = float(lambda_smooth)
        self.event_weight = float(event_weight)
        self.threshold_ratio = float(threshold_ratio)
        self.event_detector = event_detector
        self.event_quantile = float(event_quantile)
        self.time_dim = int(time_dim)
        self.eps = float(eps)

    # ------------------------- 工具函数 -------------------------

    def _to_like(self, x: torch.Tensor, ref: torch.Tensor, as_float: bool = True) -> torch.Tensor:
        """
        将张量 x 移到与 ref 相同的 device，且 dtype 设为 ref.dtype（float）或 bool。
        用于把 mask/数据对齐到 loss 的设备与精度。
        """
        dtype = ref.dtype if as_float else torch.bool
        return x.to(device=ref.device, dtype=dtype)

    def _align_mask(self, mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        """
        将 mask 对齐/广播到与 ref 相同形状。
        - 返回 float 型 mask（便于与 loss 相乘），取值通常为 0/1。
        - 若 mask 为 None，则返回 None。
        """
        if mask is None:
            return None
        # 接受 bool/float；统一转成 float 以便参与乘法
        mask = self._to_like(mask, ref, as_float=True)
        # 维度对齐：不足则在末尾补 1 维，超出则去掉多余末维
        while mask.dim() < ref.dim():
            mask = mask.unsqueeze(-1)
        while mask.dim() > ref.dim():
            mask = mask.squeeze(-1)
        # 形状不一致则按广播扩展到 ref 的形状
        if mask.shape != ref.shape:
            mask = mask.expand_as(ref)
        return mask

    def _merge_masks(
        self,
        pm: Optional[torch.Tensor],
        om: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        合并 prediction_mask 与 observed_mask：
          - 两者都有：取交集（AND）
          - 只有一个：用它
          - 都没有：返回 None
        返回 float mask（与 ref 同 device/dtype）。
        """
        pm = self._align_mask(pm, ref)
        om = self._align_mask(om, ref)
        if pm is not None and om is not None:
            vm = (pm > 0) & (om > 0)
            return vm.to(dtype=ref.dtype)
        if pm is not None:
            return (pm > 0).to(dtype=ref.dtype)
        if om is not None:
            return (om > 0).to(dtype=ref.dtype)
        return None

    @torch.no_grad()
    def _to_float_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        某些输入（例如 target）可能是整型，后续需要 torch.finfo/nanquantile 等浮点 API。
        这里确保 x 为浮点类型。
        """
        return x if x.is_floating_point() else x.float()

    @torch.no_grad()
    def _masked_max(self, x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        """
        在给定维度 dim 上计算 mask 感知的最大值（keepdim=True）。
        - 对非浮点/NaN 做防护
        - mask==0 的位置用 very_small 屏蔽，避免影响最大值
        """
        x = self._to_float_if_needed(x)
        # 将非有限值替换为 0，防止污染 max
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if mask is None:
            return x.max(dim=dim, keepdim=True).values
        # 使用 dtype 对应的极小值屏蔽无效位
        very_small = torch.finfo(x.dtype).min
        x_masked = torch.where(mask > 0, x, torch.full_like(x, very_small))
        return x_masked.max(dim=dim, keepdim=True).values

    @torch.no_grad()
    def _masked_quantile(self, x: torch.Tensor, mask: Optional[torch.Tensor], q: float, dim: int) -> torch.Tensor:
        """
        在给定维度 dim 上计算 mask 感知的分位数（keepdim=True）。
        - 先将无效位置为 NaN，再用 torch.nanquantile 忽略 NaN
        - 如果该条样本在该维上全是 NaN（全被 mask 掉），结果为 NaN，由调用方兜底
        """
        x = self._to_float_if_needed(x)
        # 将非有限值置为 NaN，便于 nanquantile 忽略
        x = torch.where(torch.isfinite(x), x, torch.nan)
        if mask is not None:
            x = torch.where(mask > 0, x, torch.nan)
        return torch.nanquantile(x, q, dim=dim, keepdim=True)

    def _get_pred_mean(self, pred) -> Optional[torch.Tensor]:
        """
        尽力从 pred 中获取“预测均值”的张量：
          - 优先使用 pred.mean
          - 否则尝试 pred.loc（部分分布以 loc 表示位置参数）
        若均无可用，返回 None（平滑项将跳过）。
        """
        mean = getattr(pred, "mean", None)
        if isinstance(mean, torch.Tensor):
            return mean
        loc = getattr(pred, "loc", None)
        if isinstance(loc, torch.Tensor):
            return loc
        return None

    # ------------------------- 核心损失计算 -------------------------

    def _loss_func(
        self,
        pred,                              # 分布对象（必须实现 .log_prob(target)，可选 .mean / .loc）
        target: torch.Tensor,              # 目标值，形状与 pred.log_prob 输出一致（通常是 [..., T]）
        prediction_mask: Optional[torch.Tensor] = None,  # 预测区掩码（1=有效，0=无效）
        observed_mask: Optional[torch.Tensor] = None,    # 观测区掩码（1=有效，0=无效）
        sample_id=None,                    # 兼容基类签名（未使用）
        variate_id=None,                   # 兼容基类签名（未使用）
        **kwargs,
    ) -> torch.Tensor:
        # 1) 基础逐点 NLL：loss_point = -log p(y | θ)
        nll = -pred.log_prob(target)  # 形状与 target 一致
        # 对可能出现的非有限值做防护（如分布与 target 域不匹配导致的 inf/-inf）
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))

        # 2) 合成有效位：prediction_mask ∩ observed_mask（若同时存在）
        valid_mask = self._merge_masks(prediction_mask, observed_mask, nll)

        # 3) 事件阈值（mask 感知）
        if self.event_detector == "quantile":
            # 分位数阈值，更鲁棒：thr = Q_q(target | valid)
            thr = self._masked_quantile(target, valid_mask, self.event_quantile, dim=self.time_dim)
            # 若全无有效点导致 thr=NaN，则回退到 ratio_max 逻辑
            fallback = self.threshold_ratio * self._masked_max(target, valid_mask, dim=self.time_dim)
            thr = torch.where(torch.isfinite(thr), thr, fallback)
        else:  # 'ratio_max'：thr = ratio * max(target | valid)
            mx = self._masked_max(target, valid_mask, dim=self.time_dim)
            thr = self.threshold_ratio * mx

        # 4) 事件加权：事件点权重为 event_weight，其余为 1
        is_event = (target > thr).to(dtype=nll.dtype)
        weight = 1.0 + (self.event_weight - 1.0) * is_event

        loss = nll * weight
        if valid_mask is not None:
            loss = loss * valid_mask  # 屏蔽无效位置

        # 5) 归一化：按有效元素个数求平均
        if valid_mask is None:
            valid_count = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
        else:
            valid_count = valid_mask.sum()

        # 若无任何有效元素，返回 0.0，避免评估器/回调出现“非法数据”告警
        if (valid_count <= 0).item():
            return loss.sum() * 0.0

        value_loss = loss.sum() / (valid_count + self.eps)

        # 6) 平滑正则：对预测的“均值轨迹”做一阶差分 L1（同样仅在有效相邻对上计算）
        smooth_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        mean = self._get_pred_mean(pred)
        if isinstance(mean, torch.Tensor):
            # 数值防护
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))

            # 在 time_dim 上做相邻差分：mean[..., t] - mean[..., t-1]
            # 这里用 index_select 避免对不定 time_dim 进行复杂切片
            idx_t   = torch.arange(1, mean.size(self.time_dim), device=mean.device)
            idx_tm1 = torch.arange(0, mean.size(self.time_dim) - 1, device=mean.device)
            diff = mean.index_select(self.time_dim, idx_t) - mean.index_select(self.time_dim, idx_tm1)

            if valid_mask is not None:
                # 仅当 (t 与 t-1) 两个位置均有效时才计入平滑项
                vm_t   = valid_mask.index_select(self.time_dim, idx_t)
                vm_tm1 = valid_mask.index_select(self.time_dim, idx_tm1)
                pair_mask = (vm_t > 0) & (vm_tm1 > 0)
                pair_count = pair_mask.sum()
                if (pair_count > 0).item():
                    smooth_loss = (torch.abs(diff) * pair_mask.to(diff.dtype)).sum() / (pair_count + self.eps)
                else:
                    smooth_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            else:
                smooth_loss = torch.mean(torch.abs(diff))

        # 7) 总损失：value_loss + λ * smooth_loss
        total_loss = value_loss + self.lambda_smooth * smooth_loss
        # 再次数值防护，确保返回有限标量
        total_loss = torch.where(torch.isfinite(total_loss), total_loss, torch.zeros_like(total_loss))
        return total_loss


# ==========================
# （可选）使用示例（仅注释示意）
# ==========================
# loss_fn = PackedValueOrientedNLLLoss(
#     lambda_smooth=0.1,
#     event_weight=2.0,
#     event_detector="quantile",   # 或 "ratio_max"
#     event_quantile=0.9,
#     threshold_ratio=0.8,
#     time_dim=-1,
# )
# total_loss = loss_fn._loss_func(
#     pred=pred_dist,                # 来自模型的分布对象（需实现 .log_prob，最好有 .mean 或 .loc）
#     target=target_tensor,          # [..., T]
#     prediction_mask=pred_mask,     # 可为 None
#     observed_mask=obs_mask,        # 可为 None
# )
