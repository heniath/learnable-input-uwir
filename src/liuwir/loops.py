"""Original epoch loops and stopping rule, extracted without numerical changes."""
import torch

def _unwrap(model):
    return model

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score
        elif (self.mode == "max" and score > self.best + self.min_delta) or (
            self.mode == "min" and score < self.best - self.min_delta
        ):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    accumulation_steps: int = 1,
):
    model.train()
    tot_loss = 0.0
    comps = {
        "l1": 0.0,
        "perceptual": 0.0,
        "ssim_loss": 0.0,
        "reconstruction": 0.0,
        "depth_smoothness": 0.0,
    }
    consecutive_amp_overflows = 0

    # BỎ TQDM, DÙNG ENUMERATE THÔNG THƯỜNG
    for batch_idx, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            uses_physics_loss = getattr(_unwrap(model), "supports_physics_loss", False)
            pred = model(inp, return_physics=True) if uses_physics_loss else model(inp)
            loss, parts = criterion(pred, gt, inp) if uses_physics_loss else criterion(pred, gt)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_idx + 1}: {loss.item()}"
            )
        scaled_loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            except RuntimeError as exc:
                if not amp_enabled:
                    raise FloatingPointError(
                        f"Non-finite gradient norm at batch {batch_idx + 1}"
                    ) from exc

                # Overflow is expected occasionally with dynamic loss
                # scaling. Skip this optimizer step and reduce the scale;
                # raising immediately prevents GradScaler from doing its job.
                consecutive_amp_overflows += 1
                old_scale = scaler.get_scale()
                bad_parameters = (
                    [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ]
                    if consecutive_amp_overflows >= 8
                    else []
                )
                scaler.update(new_scale=max(old_scale / 2.0, 1.0))
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"   [AMP] skipped batch {batch_idx + 1} after gradient overflow; "
                    f"scale {old_scale:g} -> {scaler.get_scale():g}"
                )
                if consecutive_amp_overflows >= 8:
                    names = ", ".join(bad_parameters[:12]) or "unknown"
                    raise FloatingPointError(
                        "Persistent non-finite AMP gradients after 8 scale reductions "
                        f"at batch {batch_idx + 1}; parameters: {names}"
                    ) from exc
            else:
                consecutive_amp_overflows = 0
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        tot_loss += loss.item()
        for k in comps:
            comps[k] += parts.get(k, 0.0)

    n = len(loader)
    return tot_loss / n, {k: v / n for k, v in comps.items()}


@torch.no_grad()
def val_loss_epoch(model, loader, criterion, device, amp_enabled: bool = False):
    model.eval()
    tot = 0.0
    for batch_idx, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            uses_physics_loss = getattr(_unwrap(model), "supports_physics_loss", False)
            pred = model(inp, return_physics=True) if uses_physics_loss else model(inp)
            loss, _ = criterion(pred, gt, inp) if uses_physics_loss else criterion(pred, gt)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at batch {batch_idx + 1}: {loss.item()}"
            )
        tot += loss.item()
    return tot / len(loader)
