import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class IFGSMAttack(object):
    def __init__(self, model=None, device=None,mask=None, epsilon=0.15, k=100, a=0.01):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        mask: mask for the attack_area
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.loss_fn2 = nn.L1Loss().to(device)
        self.device = device
        self.mask = mask

        # PGD(True) or I-FGSM(False)?
        self.rand = True

        #attack on specific channel?
        # self.channel = False
    
    def perturb(self, X_nat, y):
        """
        Vanilla Attack.
        """
        origin_img_src = X_nat.clone().detach_()#保留原始的img_src
        origin_img_src = origin_img_src.to(self.device)

        if self.rand: # PGD attack
            X_nat = X_nat.to(self.device)
            random = torch.rand_like(origin_img_src).uniform_(-self.epsilon,self.epsilon).to(self.device)
            x_tmp = X_nat + random
            X_nat = x_tmp.clone().detach_()

        # X_nat.to('cpu')
        
        # self.model.set_input(X_nat)#设置model的数据

        for i in range(self.k):
            X_nat.requires_grad_(True)
            # self.model.img_src.requires_grad = True #对img_src求梯度
            # self.model.forward()

            output = self.model.extract(X_nat) #攻击对象：style_code

            self.model.zero_grad()

            # Minus in the loss means "towards" and plus means "away from"
            # use mse loss
            loss = self.loss_fn(output, y)

            # loss = ((output - y)**2).sum() #self_defined loss
            # loss = loss.mean()

            loss.requires_grad_(True) #!!解决无grad bug
            loss.backward()
            grad = X_nat.grad.data

            img_src_adv = X_nat + self.a * torch.sign(grad)

            # eta = torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon)#加入的噪声
            # 对batch 做 mean
            eta = torch.mean(torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon).detach_(),dim=0)#加入的噪声

            X_nat = torch.clamp(origin_img_src + eta, min=-1, max=1).detach_()#攻击后的img_src结果

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None
        #返回攻击后的img_src和noise
        return X_nat, eta 
