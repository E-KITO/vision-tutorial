import torch
import torch.nn as nn
import sys
sys.path.append('./src/my_project')
from model import Resblock, get_ResNet152


if __name__ == "__main__":
    def main():
        net = get_ResNet152(Resblock, 1000)
        y = net(torch.randn(10, 3, 224, 224))
        print(y.size())
    
    main()