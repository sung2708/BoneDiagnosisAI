import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from ..configs import MODEL_CONFIG

dim = MODEL_CONFIG.dim
device = MODEL_CONFIG.device
train_image_file = MODEL_CONFIG.train_image_file
valid_image_file = MODEL_CONFIG.valid_image_file
test_image_file = MODEL_CONFIG.test_image_file

class Transfer(nn.Module):
    def __init__(self, model=models.resnet50(pretrained=True), trans=transforms.Compose([transforms.Resize((224,224)), transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),transforms.RandomRotation(5),transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        super(Transfer, self).__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad=False
        self.trans = trans
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(1024, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(512, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Conv2d(256, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(64, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, name):
        v_list = []
        for v in name:
            if os.path.isfile(train_image_file + v + '.jpg') == True:
                f = Image.open(train_image_file + v + '.jpg')
            elif os.path.isfile(valid_image_file + v + '.jpg') == True:
                f = Image.open(valid_image_file + v + '.jpg')
            else:
                f = Image.open(test_image_file + v + '.jpg')
            imag = self.trans(f)
            if len(imag) == 1:
                imag = torch.cat([imag,imag,imag],dim=0)
            v_list.append(imag)
            f.close()
        v_list = torch.stack(v_list).to(device)
        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        v_2 = self.gap2(self.relu(self.conv2(fix2(v_list)))).view(-1,dim)
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.relu(self.conv3(fix3(v_list)))).view(-1,dim)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.relu(self.conv4(fix4(v_list)))).view(-1,dim)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.relu(self.conv5(fix5(v_list)))).view(-1,dim)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.relu(self.conv7(fix7(v_list)))).view(-1,dim)
        return v_2, v_3, v_4, v_5, v_7
