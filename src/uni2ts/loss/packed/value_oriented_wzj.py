from uni2ts.loss.packed import PackedPointLoss
import torch

class PackedValueOrientedMAELoss(PackedPointLoss):
    def __init__(self, lambda_smooth: float = 0.1, event_weight: float = 2.0):
        """
        lambda_smooth: 平滑正则化权重
        event_weight: 关键事件权重（关键区间乘以该权重，其它为1.0）
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.event_weight = event_weight

    def forward(
        self,
        pred: torch.Tensor,   # [*batch, horizon]
        target: torch.Tensor,
        prediction_mask: torch.Tensor = None,
        observed_mask: torch.Tensor = None,
        sample_id: torch.Tensor = None,
        variate_id: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Event mask逻辑：关键区间自动加权，其它为1
        # 例：目标大于阈值为关键事件（你可自定义！）
        threshold = 0.8 * target.max(dim=-1, keepdim=True)[0]
        is_event = (target > threshold).float()
        weight = 1.0 + (self.event_weight - 1.0) * is_event    # 关键区间加权

        # 预测与目标之差
        loss = weight * torch.abs(pred - target)

        # 如果有mask（只计算预测区间/有效观测），mask掉无关部分
        if prediction_mask is not None:
            loss = loss * prediction_mask
        if observed_mask is not None:
            loss = loss * observed_mask

        # 均值损失
        value_loss = loss.sum() / (loss.numel() if (prediction_mask is None) else (prediction_mask.sum() + 1e-8))

        # 平滑性正则项
        diff = pred[..., 1:] - pred[..., :-1]
        smooth_loss = torch.mean(torch.abs(diff))
        total_loss = value_loss + self.lambda_smooth * smooth_loss
        return total_loss
