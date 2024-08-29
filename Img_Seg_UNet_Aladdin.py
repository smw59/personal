 # use padded conv to simplfy it 
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(DoubleConv, self).__init__()
    self.conv = nn.Segquential(
    nn.conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU(inplace=True),
    nn.conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU(inplace=True),
    )
  def forward(self, x):
    return self.conv(x)
