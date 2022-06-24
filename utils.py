import os
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer, required

## CosineAnnealingWarmUpRestarts
import math
from torch.optim.lr_scheduler import _LRScheduler

def device_as(t1, t2):
    """Moves tensor1 (t1) to the device of tensor2 (t2)"""
    return t1.to(t2.device)

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
            

## lr schedulers
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.gamma)
    elif opt.lr_policy == 'multi-step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(opt.epoch/2), int(opt.epoch * 0.75)], gamma=opt.gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        # 'min' : metric이 감소를 멈출 때 lr update (val loss에 적용시에 사용)
        # factor: lr를 감소시키는 비율 lr * factor
        # threshold: 새로운 optimum이 될 수 있는 threshold (얼마 차이나면 optimum update되었다고 볼 수 있나?)
        # patience: metric이 향상 안될 때, 몇 epoch을 참을 것인가?
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0) 
        # eta_min : minimum # T_max : 첫번째 최소점을 찍는 epoch
    elif opt.lr_policy == 'SGDR':
        # warmup을 위해 초기 시작 lr 을 0으로 지정.
        # https://gaussian37.github.io/dl-pytorch-lr_scheduler/ 참고
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0 
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=opt.lr, T_up=10, gamma=opt.gamma)
        
        # T_0 : 최초 주기값 (T_mult에 의해 점점 주기가 커짐)
        # T_mult : 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값에 해당
        # eta_max : lr의 최대값 (warmup 시 eta_max까지 값이 증가)
        # T_up : Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정
        # gamma : lr를 얼마나 줄여나갈 것인지.(eta_max에 곱해지는 스케일값)
    elif opt.lr_policy == 'None':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler