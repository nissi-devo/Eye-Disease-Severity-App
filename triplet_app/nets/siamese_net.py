import torch
from torch import nn
from torchvision import transforms, models

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = models.resnet101(pretrained=True)
        self.cnn1.fc = nn.Linear(2048, 3)  # mapping input image to a 3 node output

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3