from uni2ts.loss.packed import PackedDistributionLoss
import torch

class PackedValueOrientedNLLLoss(PackedDistributionLoss):
    """
    Value-oriented weighted NLL loss for distributional time series forecasting,
    with event region emphasis and smoothness regularization.
    """

    def __init__(self, lambda_smooth: float = 0.1, event_weight: float = 2.0, threshold_ratio: float = 0.8):
        """
        :param lambda_smooth: Weight for smoothness regularization
        :param event_weight: Weight for event region
        :param threshold_ratio: Threshold ratio for identifying key events
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.event_weight = event_weight
        self.threshold_ratio = threshold_ratio

    def _align_mask(self, mask, ref):
        """Align mask dimensions with reference tensor."""
        if mask is None:
            return None
        # Expand dims if needed
        while mask.dim() < ref.dim():
            mask = mask.unsqueeze(-1)
        # Squeeze if too many dims (should not often happen, but for safety)
        while mask.dim() > ref.dim():
            mask = mask.squeeze(-1)
        # Expand to shape if still not match
        if mask.shape != ref.shape:
            mask = mask.expand_as(ref)
        return mask

    def _loss_func(
        self, pred, target, prediction_mask=None, observed_mask=None, **kwargs
    ):
        # 负对数似然损失
        loss = -pred.log_prob(target)

        # Event weighting (value-oriented)
        threshold = self.threshold_ratio * target.max(dim=-1, keepdim=True)[0]
        is_event = (target > threshold).float()
        weight = 1.0 + (self.event_weight - 1.0) * is_event
        loss = loss * weight

        # Align masks
        prediction_mask = self._align_mask(prediction_mask, loss)
        observed_mask = self._align_mask(observed_mask, loss)

        # Apply masks
        if prediction_mask is not None:
            loss = loss * prediction_mask
        if observed_mask is not None:
            loss = loss * observed_mask

        # Normalization: count valid elements
        valid_count = loss.numel()
        if prediction_mask is not None:
            valid_count = prediction_mask.sum()
        elif observed_mask is not None:
            valid_count = observed_mask.sum()
        value_loss = loss.sum() / (valid_count + 1e-8)

        # Smoothness penalty (for predicted mean)
        if hasattr(pred, "mean"):
            diff = pred.mean[..., 1:] - pred.mean[..., :-1]
            smooth_loss = torch.mean(torch.abs(diff))
        else:
            smooth_loss = 0.0

        total_loss = value_loss + self.lambda_smooth * smooth_loss
        return total_loss
