import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(torch.nn.Module):     # 繼承 torch 的 Module
    def __init__(self):
        super(Classifier, self).__init__()     # 繼承 __init__ 功能
        self.hidden1 = torch.nn.Linear(256, 128)   # 隱藏層線性輸出
        self.hidden2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 2)       # 輸出層線性輸出
 
    def forward(self, x):
        # 正向傳播輸出值,神經網路輸出值
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))# 激勵函數(隱藏層的線性值)
        x = self.out(x)
        return x