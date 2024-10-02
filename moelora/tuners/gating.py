import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
NEG_INF = -5e4

DM_PARAM_EXCEPTION: str = "Expecting argument `{}` to be positive float, get {}"
ALL_ID_IGN: str = "All `role_ids` are larger than `num_moe`, " \
                  "so the gating loss will not be calculated for this batch"

warnings.filterwarnings("always", ALL_ID_IGN, RuntimeWarning)

def enable_grad(module, input, output):
    output.requires_grad_(True)


class Dense(nn.Module):
    def __init__(self, dim: int, num_moe: int, **kwargs) -> None:
        super().__init__()
        self.dim = 1024
        self.num_moe = num_moe
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)

        if self.training:
            self._grad_hook = self.linear_layer.register_forward_hook(enable_grad)
            # self._back_hook = self.linear_layer.register_full_backward_hook(back_grad)

    def forward(self, x):
        logits = self.linear_layer(x)

        return None, logits

class DenseWithMask(Dense):
    def __init__(self, dim: int, num_moe: int, **kwargs) -> None:
        super().__init__(dim, num_moe, **kwargs)
        # init params loss_alpha
        loss_alpha = kwargs.get("loss_alpha", None)
        assert (isinstance(loss_alpha, float) and loss_alpha > 0.0), \
            DM_PARAM_EXCEPTION.format("loss_alpha", loss_alpha)
        self.loss_alpha = loss_alpha
    
        # init params noise_norm
        noise_norm = kwargs.get("noise_norm", None)
        assert (isinstance(loss_alpha, float) and loss_alpha >= 0.0), \
            DM_PARAM_EXCEPTION.format("noise_norm", noise_norm)
        self.noise_norm = noise_norm
        if self.noise_norm == 0.0:
            warnings.warn("`noise_norm` is set to 0.0, will not add noise to gating logits")
        
        self.role_ids = []
        
    def forward(self, x):
        logits = self.linear_layer(x)
        loss = None

        if len(self.role_ids) == 0:
            warnings.warn(
                "WARN: argument `role_ids` not specified, reducing to `Dense` behavior"
            )

            return loss, logits
        
        # add noise
        right_batch = self.role_ids[0] < self.num_moe
        if self.training:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
            if not torch.any(right_batch):
                # create a dummy loss
                loss = 0.0 * loss_fn(logits, F.softmax(logits.detach(), dim=-1))
            else:
                logits_for_loss = logits[right_batch, :]
                noise =  self.noise_norm * torch.randn_like(logits_for_loss).clamp_(-2*self.noise_norm, 
                                                                        2*self.noise_norm)
                
                loss = self.loss_alpha * loss_fn(logits_for_loss + noise, self.role_ids[0])

        mask_neg = torch.full_like(logits, False).to(device=logits.device, dtype=torch.bool)
        # 如果有role_ids超过num_moe就原样复制过去
        mask_neg[right_batch, self.role_ids[0][right_batch]] = True
        # 在wrap_model中加入static_graph=True以后可以不detach, 否则最后一层会报二次调用exception
        logits = logits.masked_fill(mask_neg, NEG_INF)

        return loss, logits

GATING_TO_MODEL_MAPPING = {
    "Dense": Dense,
    "DenseWithMask": DenseWithMask,
}