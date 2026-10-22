"""Historical cosine schedule; pilot uses its first 30 of 100 epochs."""
import math
from torch.optim.lr_scheduler import _LRScheduler

def _get_position_from_periods(iteration, cumulative_period):
    """Return the index of the right-closest period boundary."""
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


class CosineAnnealingRestartLR(_LRScheduler):
    """
    Cosine annealing with periodic restarts and a single global eta_min.

    Args:
        periods        (list[int])   : Length of each restart cycle.
        restart_weights(list[float]) : LR scale at the start of each cycle.
        eta_min        (float)       : Global minimum learning rate.
    """

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1):
        assert len(periods) == len(restart_weights), (
            "periods and restart_weights must have the same length"
        )
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        self.cumulative_period = [sum(periods[: i + 1]) for i in range(len(periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = _get_position_from_periods(self.last_epoch, self.cumulative_period)
        weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        period = self.periods[idx]
        progress = (self.last_epoch - nearest_restart) / period
        return [
            self.eta_min
            + weight * 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
