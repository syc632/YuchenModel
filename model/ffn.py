import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FFNConfig:
    d_latent:int =128
    d_model: int  = 512
    d_inner:int = 768



class SiTUGLU(nn.Module):
    def __init__(self,d_model,d_inner):
        super().__init__()
        self.beta = 4
        self.linear_beta = 25
        self.W_up = nn.Linear(d_model,d_inner,bias=False)
        self.W_gate = nn.Linear(d_model,d_inner,bias=False)
        self.W_down = nn.Linear(d_inner,d_model,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()
    def forward(self,x):
        residual = x
        gate = self.W_gate(x)
        x = self.W_up(x)
        gate = self.beta*(self.tanh_1(gate/self.beta)*self.sigmoid(gate))
        x = self.linear_beta*(self.tanh_2(x/self.linear_beta))
        output = x*gate
        output = self.W_down(output)
        output = output + residual
        return output


class SwiGlu(nn.Module):
    def __init__(self,d_model,d_inner):
        super().__init__()
        self.W_up = nn.Linear(d_model,d_inner,bias=False)
        self.W_down = nn.Linear(d_inner,d_model,bias=False)
        self.gate = nn.Linear(d_model,d_inner)
        self.act = nn.SiLU()
    def forward(self,x):
        residual = x
        gate = self.gate(x)
        x = self.act(self.W_up(x))
        output = x*gate
        output = self.W_down(output)
        output = output + residual
        return output
if __name__ == "__main__":
    x = torch.randn((1,32,128))
    cfg = FFNConfig()
    a = SwiGlu(cfg)
    b = SiTUGLU(cfg)
    print(b(x).shape)
    print(a(x).shape)