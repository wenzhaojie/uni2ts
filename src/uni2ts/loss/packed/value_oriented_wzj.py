import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.loss.packed import PackedDistributionLoss  # 如果路径不同，请按你的工程实际调整


class PackedValueOrientedNLLLoss(PackedDistributionLoss):
    """
    价值导向的 NLL：
      - 对“突增点”（|Δtarget| 超过全局分位数阈值）放大 NLL 权重
      - 加入对 pred.mean 的一阶差分平滑正则 (L2)
    返回逐点损失 (*batch seq_len #dim)，与 PackedLoss.reduce_loss 完整兼容
    """

    def __init__(
        self,
        spike_quantile: float = 0.9,      # 突增阈值分位数（全局）
        spike_weight: float = 3.0,        # 突增点额外权重 (>1 放大)
        smooth_lambda: float = 0.1,       # 平滑正则系数 (>=0)
        apply_only_on_pred_mask: bool = True,  # 突增识别是否仅在预测区间内进行
        detach_threshold: bool = True,    # 量化阈值计算是否与梯度图断开
        eps: float = 1e-8,                # 数值稳定项
    ):
        super().__init__()
        assert 0.0 < spike_quantile < 1.0, "spike_quantile must be in (0,1)"
        assert spike_weight >= 1.0, "spike_weight should be >= 1.0"
        assert smooth_lambda >= 0.0, "smooth_lambda should be >= 0.0"
        self.spike_quantile = spike_quantile
        self.spike_weight = spike_weight
        self.smooth_lambda = smooth_lambda
        self.apply_only_on_pred_mask = apply_only_on_pred_mask
        self.detach_threshold = detach_threshold
        self.eps = eps

    def _loss_func(
        self,
        pred: Distribution,
        target: Float[Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[Tensor, "*batch seq_len"],
        observed_mask: Bool[Tensor, "*batch seq_len #dim"],
        sample_id: Int[Tensor, "*batch seq_len"],
        variate_id: Int[Tensor, "*batch seq_len"],
    ) -> Float[Tensor, "*batch seq_len #dim"]:
        device = target.device
        # -------------------------
        # 1) 基础 NLL (逐点)
        # -------------------------
        base_nll = -pred.log_prob(target)  # 形状: *batch seq_len #dim
        # 避免极端分布返回 inf/NaN
        base_nll = torch.nan_to_num(base_nll, nan=0.0, posinf=1e6, neginf=0.0)

        # -------------------------
        # 2) 突增点识别与加权
        #    使用 target 的一阶差分 |Δy_t|，把 Δ 归到“当前时刻 t”
        # -------------------------
        seq_dim = -2  # 约定: 倒数第二维是 seq_len，最后一维是 #dim
        # 有效观测的成对掩码：t 与 t-1 都有观测（逐维）
        obs_t = observed_mask
        obs_tm1 = torch.zeros_like(obs_t)
        obs_tm1[..., 1:, :] = obs_t[..., :-1, :]
        pair_obs_mask = obs_t & obs_tm1  # *batch seq_len #dim, t=0 处为 False

        # 若只在预测区间内寻找突增，额外与 pred_mask 相交
        if self.apply_only_on_pred_mask:
            pred_mask_exp = prediction_mask.unsqueeze(-1).expand_as(target)
            pred_mask_tm1 = torch.zeros_like(pred_mask_exp)
            pred_mask_tm1[..., 1:, :] = pred_mask_exp[..., :-1, :]
            pair_obs_mask = pair_obs_mask & pred_mask_exp & pred_mask_tm1

        # 计算 |Δtarget_t|，t=0 处填 0
        delta = torch.zeros_like(target)
        delta[..., 1:, :] = (target[..., 1:, :] - target[..., :-1, :]).abs()

        # 只用有效对的 delta 值计算全局分位数阈值
        valid_delta = delta[pair_obs_mask]
        if valid_delta.numel() > 0:
            if self.detach_threshold:
                thresh = torch.quantile(valid_delta.detach(), self.spike_quantile)
            else:
                thresh = torch.quantile(valid_delta, self.spike_quantile)
        else:
            # 没有有效对，阈值置为 +∞（即不加权）
            thresh = torch.tensor(float("inf"), device=device, dtype=target.dtype)

        # spike_mask：当前时刻 t 的 |Δy_t| 是否超过阈值（且满足 pair_obs_mask）
        spike_mask = (delta > thresh) & pair_obs_mask  # *batch seq_len #dim

        # 构造逐点权重
        spike_w = torch.ones_like(base_nll)
        spike_w = spike_w + (self.spike_weight - 1.0) * spike_mask.to(base_nll.dtype)

        # -------------------------
        # 3) 平滑正则（分布均值的 TV-L2： (μ_t - μ_{t-1})^2 ）
        #    将对 (t,t-1) 的惩罚平均分摊回两个时刻，使其成为“逐点”项
        # -------------------------
        reg_per_token = torch.zeros_like(base_nll)
        if self.smooth_lambda > 0.0:
            # 预测均值：与 target 同形状
            mu = pred.mean
            mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

            # 计算 μ 的一阶差分
            dmu = torch.zeros_like(mu)
            dmu[..., 1:, :] = mu[..., 1:, :] - mu[..., :-1, :]

            # 平滑正则的成对掩码：只在“预测∧观测”的配对上生效
            pred_mask_exp = prediction_mask.unsqueeze(-1).expand_as(mu)
            pred_mask_tm1 = torch.zeros_like(pred_mask_exp)
            pred_mask_tm1[..., 1:, :] = pred_mask_exp[..., :-1, :]

            # 需要两个时刻都在“预测区”且两个点都有观测
            pair_mask_reg = (
                pred_mask_exp
                & pred_mask_tm1
                & obs_t
                & obs_tm1
            )

            # 逐对的正则值 (t>0 处)：(μ_t - μ_{t-1})^2
            reg_pair = (dmu ** 2) * pair_mask_reg.to(mu.dtype)

            # 把每一对的正则值平摊到 t 与 t-1 两个时刻上（各 1/2）
            # 注意：对 t=0，没有对，因此不加
            half = 0.5 * self.smooth_lambda
            # 加到 t 位置
            reg_per_token[..., 1:, :] += half * reg_pair[..., 1:, :]
            # 加到 t-1 位置
            reg_per_token[..., :-1, :] += half * reg_pair[..., 1:, :]

        # -------------------------
        # 4) 合成总逐点损失：加权 NLL + 平滑正则
        # -------------------------
        loss = base_nll * spike_w + reg_per_token
        return loss
