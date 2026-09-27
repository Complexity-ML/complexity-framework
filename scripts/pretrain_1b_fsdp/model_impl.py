"""FSDP root wrapper keeping the tied output weight live through fused loss."""
from torch import nn
from complexity.models import ComplexityModel
from complexity.core.losses.fused_ce import fused_linear_causal_lm_loss, has_liger_fused_linear_ce

class TrainingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ComplexityModel(config)
        self.model.gradient_checkpointing_enable()

    def forward(self, x, y):
        hidden = self.model(x, return_logits=False)['last_hidden_state']
        loss, _ = fused_linear_causal_lm_loss(
            hidden, self.model.get_output_embeddings().weight, y,
            use_liger=True, sync_metrics=False,
        )
        return loss


