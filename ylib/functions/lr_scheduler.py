import numpy as np

# Cosine Schedulerの実装
class CosineScheduler:
    def __init__(self, num_epoch, lr, warmup_length):
        """
        引数
        num_epoch: 学習エポック数
        lr: 学習率
        warmup_length: warmup適用エポック数
        """
        self.num_epoch = num_epoch
        self.lr = lr
        self.warmup = warmup_length
    
    def __call__(self, epoch):
        """
        引数
        epoch: 現在のエポック数
        """
        progress = (epoch - self.warmup) / (self.num_epoch - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1. + np.cos(np.pi * progress))

        if self.warmup:
            lr = lr * min(1., (epoch+1) / self.warmup)

        return lr

# 学習率の更新を行う関数
def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr