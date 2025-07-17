from uni2ts.loss.packed import PackedPointLoss
import torch

class PackedValueOrientedMAELoss(PackedPointLoss):
    def __init__(self, lambda_smooth: float = 0.1, event_weight: float = 2.0, threshold_ratio: float = 0.8):
        """
        lambda_smooth: 平滑正则化权重
        event_weight: 关键事件权重（关键区间乘以该权重，其它为1.0）
        threshold_ratio: 用于检测“关键事件”的阈值比例
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.event_weight = event_weight
        self.threshold_ratio = threshold_ratio

    def _loss_func(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor = None,
        observed_mask: torch.Tensor = None,
        sample_id: torch.Tensor = None,
        variate_id: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # 兼容分布或张量
        if hasattr(pred, "mean"):
            pred_tensor = pred.mean
        else:
            pred_tensor = pred

        # Event加权mask逻辑：可自定义
        threshold = self.threshold_ratio * target.max(dim=-1, keepdim=True)[0]  # [batch, seq, 1] or [batch, seq]
        is_event = (target > threshold).float()
        weight = 1.0 + (self.event_weight - 1.0) * is_event

        loss = weight * torch.abs(pred_tensor - target)   # [batch, seq, ...patch] or [batch, seq]

        # 保证 mask 形状和 loss 一致
        def align_mask(mask, ref):
            if mask is None:
                return None
            while mask.dim() < ref.dim():
                mask = mask.unsqueeze(-1)
            if mask.shape != ref.shape:
                mask = mask.expand_as(ref)
            return mask

        prediction_mask = align_mask(prediction_mask, loss)
        observed_mask = align_mask(observed_mask, loss)

        if prediction_mask is not None:
            loss = loss * prediction_mask
        if observed_mask is not None:
            loss = loss * observed_mask

        # 避免除0
        valid_count = loss.numel()
        if prediction_mask is not None:
            valid_count = prediction_mask.sum()
        elif observed_mask is not None:
            valid_count = observed_mask.sum()
        value_loss = loss.sum() / (valid_count + 1e-8)

        # 平滑性正则项 (只对时间维/patch维)
        # 假设最后一维是patch，如果没有patch就是对seq维
        diff = pred_tensor[..., 1:] - pred_tensor[..., :-1]
        smooth_loss = torch.mean(torch.abs(diff))

        total_loss = value_loss + self.lambda_smooth * smooth_loss
        return total_loss
