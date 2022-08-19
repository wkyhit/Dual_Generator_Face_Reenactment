import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class IFGSMAttack(object):
    def __init__(self, model=None, device=None,mask=None, epsilon=0.1, k=100, a=0.01):
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
        self.channel = True
    
    def _batch_multiply_tensor_by_vector(self,vector, batch_tensor):
        """Equivalent to the following
        for ii in range(len(vector)):
            batch_tensor.data[ii] *= vector[ii]
        return batch_tensor
        """
        return (
            batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

    def batch_multiply(self,float_or_vector, tensor):
        if isinstance(float_or_vector, torch.Tensor):
            assert len(float_or_vector) == len(tensor)
            tensor = self._batch_multiply_tensor_by_vector(float_or_vector, tensor)
        elif isinstance(float_or_vector, float):
            tensor *= float_or_vector
        else:
            raise TypeError("Value has to be float or torch.Tensor")
        return tensor

    def perturb(self, X_nat, y, reference_lt):
        """
        Vanilla Attack.
        """
        origin_img_src = X_nat.clone().detach_()#保留原始的img_src
        origin_img_src = origin_img_src.to(self.device)
        y = y.to(self.device)
        reference_lt = reference_lt.to(self.device)

        if self.rand: # PGD attack
            X_nat = X_nat.to(self.device)
            random = torch.rand_like(origin_img_src).uniform_(-self.epsilon,self.epsilon).to(self.device)
            x_tmp = X_nat + random
            x_tmp = torch.clamp(x_tmp, min=0, max=1).detach_()#攻击后的img_src结果
            X_nat = x_tmp.clone().detach_()


        X_nat = X_nat.to(self.device)        

        for i in range(self.k):
            #注意要将solver.extract中的注解no_grad()注释掉！！！
            X_nat.requires_grad = True 
            # self.model.img_src.requires_grad = True #对img_src求梯度
            # self.model.forward()

            style_code = self.model.extract(X_nat) #先提取style_code
            _,_,_,_,output = self.model.sample(style_code,reference_lt) #攻击对象：最终生成的It

            # self.model.zero_grad() #梯度清零需不需要？
            self.model.nets_ema.style_encoder.zero_grad() #梯度清零?
            self.model.nets_ema.generator.zero_grad() #梯度清零?
            
            #对单通道的梯度mask
            if self.channel:
                channel_idx = 2 #通道2噪声不明显
                grad_channel_mask = torch.zeros_like(X_nat)
                grad_channel_mask[:,channel_idx,:,:] = 1
                grad_channel_mask = grad_channel_mask.to(self.device)

            # Minus in the loss means "towards" and plus means "away from"
            
            # 1- use mse loss
            # loss = self.loss_fn(output, y)
            
            # 2- use l1 loss
            # loss = self.loss_fn2(output, y)

            # 3- self_defined loss（mse loss）
            # loss = ((output - y)**2).sum() 
            # loss = loss.mean()

            # 4- nullfying attack (mse loss)
            loss = -1*((output-origin_img_src)**2).sum()
            loss = loss.mean()
            #or
            # loss = -1*self.loss_fn(output, origin_img_src)

            loss.requires_grad_(True) #!!解决无grad bug
            loss.backward()
            grad = X_nat.grad.data


            #******基于 L infinity*******
            # if self.channel:
            #     grad = grad * grad_channel_mask
            # img_src_adv = X_nat + self.a * torch.sign(grad) #l Infinity attack

            #*****基于L2 attack*******
            batch_size = grad.size(0)
            p = 2 # L2
            samll_constant = 1e-6 #防止梯度为0的情况
            norm = grad.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
            norm = torch.max(norm, torch.ones_like(norm)*samll_constant)
            grad = self.batch_multiply(1./norm, grad)
            if self.channel:
                grad = grad * grad_channel_mask
            img_src_adv = X_nat + self.a * grad

            #此写法会出现梯度为0的情况！！！
            # img_src_adv = X_nat + grad/grad.norm(p=2,dim=0,keepdim=True) * self.a 

            eta = torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon)#加入的噪声
            # 对batch 做 mean
            # eta = torch.mean(torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon).detach_(),dim=0)#加入的噪声
            #注意tensor取值0~1
            X_nat = torch.clamp(origin_img_src + eta, min=0, max=1).detach_()#攻击后的img_src结果

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None
        #返回攻击后的img_src和noise
        # return X_nat, eta 
        return X_nat, X_nat-origin_img_src

