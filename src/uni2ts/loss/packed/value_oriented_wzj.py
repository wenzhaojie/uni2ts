from uni2ts.loss.packed import PackedDistributionLoss
import torch

class PackedValueOrientedNLLLoss(PackedDistributionLoss):
    def __init__(self, lambda_smooth: float = 0.1, event_weight: float = 2.0, threshold_ratio: float = 0.8):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.event_weight = event_weight
        self.threshold_ratio = threshold_ratio

    def _loss_func(
        self, pred, target, prediction_mask=None, observed_mask=None, sample_id=None, variate_id=None, **kwargs
    ):
        loss = -pred.log_prob(target)

        threshold = self.threshold_ratio * target.max(dim=-1, keepdim=True)[0]
        is_event = (target > threshold).float()
        weight = 1.0 + (self.event_weight - 1.0) * is_event

        loss = loss * weight

        # Mask shape alignment
        if prediction_mask is not None:
            prediction_mask = prediction_mask.unsqueeze(-1).expand_as(loss)
            loss = loss * prediction_mask

        if observed_mask is not None:
            observed_mask = observed_mask.unsqueeze(-1).expand_as(loss)
            loss = loss * observed_mask

        valid_count = loss.numel()
        if prediction_mask is not None:
            valid_count = prediction_mask.sum()
        elif observed_mask is not None:
            valid_count = observed_mask.sum()

        value_loss = loss.sum() / (valid_count + 1e-8)

        # Smooth regularization for distribution mean
        if hasattr(pred, 'mean'):
            diff = pred.mean[..., 1:] - pred.mean[..., :-1]
            smooth_loss = torch.mean(torch.abs(diff))
        else:
            smooth_loss = 0.0

        total_loss = value_loss + self.lambda_smooth * smooth_loss
        return total_loss
