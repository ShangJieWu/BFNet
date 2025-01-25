import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_bce_loss(inputs, targets):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
    wbce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inputs = torch.sigmoid(inputs)
    inter = ((inputs * targets) * weit).sum(dim=(2, 3)) 
    union = ((inputs + targets) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def bce_loss(inputs, targets):
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return bce.mean()


def dice_loss(inputs, targets):
    inputs = torch.sigmoid(inputs)
    inter = (inputs * targets).sum(dim=(2, 3))
    union = (inputs + targets).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (union + 1)
    return dice.mean()


def get_metrics(pred, mask):
    pred = (pred > 0.5).float()
    pred_positives = pred.sum(dim=(2, 3))
    mask_positives = mask.sum(dim=(2, 3))
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).float().mean(dim=(2, 3))
    recall = inter / (mask_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred - mask)).mean(dim=(2, 3))

    return dice, iou, acc, recall, f2, mae


class SmoothedValue:
    """
    Track a series of value and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.
        self.count = 0.
        self._global_max = 0.
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        if value > self._global_max:
            self._global_max = value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def global_max(self):
        return self._global_max

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise Exception
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def __str__(self):
        log_str = []
        for name, meter in self.meters.items():
            log_str.append(
                "{}: {}".format(name, meter)
            )
        return self.delimiter.join(log_str)
