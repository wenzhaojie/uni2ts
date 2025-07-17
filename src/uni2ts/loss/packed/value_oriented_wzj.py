from uni2ts.loss.packed import PackedPointLoss
import torch

class PackedValueOrientedMAELoss(PackedPointLoss):
    def __init__(self, lambda_smooth: float = 0.1, event_weight: float = 2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.event_weight = event_weight

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
        # Event mask 生成逻辑
        threshold = 0.8 * target.max(dim=-1, keepdim=True)[0]
        is_event = (target > threshold).float()
        weight = 1.0 + (self.event_weight - 1.0) * is_event

        loss = weight * torch.abs(pred - target)
        if prediction_mask is not None:
            loss = loss * prediction_mask
        if observed_mask is not None:
            loss = loss * observed_mask

        # 避免除0
        norm = loss.numel() if prediction_mask is None else (prediction_mask.sum() + 1e-8)
        value_loss = loss.sum() / norm

        # 平滑正则项
        diff = pred[..., 1:] - pred[..., :-1]
        smooth_loss = torch.mean(torch.abs(diff))

        total_loss = value_loss + self.lambda_smooth * smooth_loss
        return total_loss

