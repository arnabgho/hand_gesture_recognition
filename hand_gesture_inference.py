import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [ transforms.Scale([64,64],Image.BICUBIC),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='../data/hands/train/',
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='../data/hands/val/',
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('fu', 'metal', 'ok_hand', 'point_up',
           'punch', 'raised_hand', 'shaka', 'thumbs_up', 'victory')


nc = 3
ndf = 8

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 9, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)


net = Net()

net.load_state_dict(torch.load('hands_trained.pth'))

def inference(img_name):
    pil_img = Image.open(img_name).convert('RGB')
    im = transform(pil_img)
    A = im.resize_(1,nc,64,64)
    output = net(A)
    _, predicted = torch.max(output.data, 1)
    print(classes[predicted.data])


inference('test.jpg')

