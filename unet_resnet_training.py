import copy
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import pandas as pd
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

data_transforms = {
    'train':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    'validation':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    'testing':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
}

image_datasets = {
    'train': datasets.ImageFolder('final_data_set/training_cropped_faces/training_resized_cropped_images',
                                  data_transforms['train']),
    'validation': datasets.ImageFolder('final_data_set/validation_cropped_faces/validation_resized_cropped_images',
                                       data_transforms['validation']),
    'testing': datasets.ImageFolder('final_data_set/testing_cropped_faces/testing_resized_cropped_images',
                                    data_transforms['testing'])
}


def reverse_key_value(image_dataset_dictionary):
    return dict((v, k) for k, v in image_dataset_dictionary.items())


class_to_index = {'train': reverse_key_value(image_datasets['train'].class_to_idx),
                  'validation': reverse_key_value(image_datasets['validation'].class_to_idx),
                  'testing': reverse_key_value(image_datasets['testing'].class_to_idx)}

# print(class_to_index)

dataloaders = {
    'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=8,
                                    shuffle=True,
                                    num_workers=os.cpu_count()),
    'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=8,
                                    shuffle=False,
                                    num_workers=os.cpu_count()),
    'testing':
        torch.utils.data.DataLoader(image_datasets['testing'],
                                    batch_size=8,
                                    shuffle=False,
                                    num_workers=os.cpu_count())
}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        if self.bilinear:  # if bilinear, use normal convolutions to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(self.in_channels, self.out_channels, self.in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(self.in_channels, self.out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.Sigmoid):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            out = self.act(x)
        else:
            out = x
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, act=nn.Sigmoid):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // self.factor)
        self.up1 = Up(1024, 512 // self.factor, self.bilinear)
        self.up2 = Up(512, 256 // self.factor, self.bilinear)
        self.up3 = Up(256, 128 // self.factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes, act=act)

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class UResnetModel(nn.Module):
    def __init__(self, num_of_classes, n_channels, writer):
        super(UResnetModel, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_of_classes
        self.unet = UNet(n_channels, 3)
        self.writer = writer
        self.base_model = models.resnet50(pretrained=True)
        self.resnet_fc_in_features = self.base_model.fc.in_features

        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(self.resnet_fc_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        ).to(device)

    def generateNoise(self, size):
        return torch.randn(size) * 0.1

    def forward(self, x):
        b, c, w, h = x.shape  # batch_size, fc_input_size, width, height

        noise = (self.generateNoise(x.size())).to(device)

        mask = self.unet(x)
        unmask = 1 - mask

        x = (x * mask) + (unmask * noise)

        x = self.base_model(x)
        x = x.reshape(b, -1)
        x = self.classifier(x)

        return x, mask, unmask


# def log_grid_image(labels, phase, logger, im, iter, nrow=int(math.ceil(10 ** 0.5))):
#     print(im.shape)
#     print(labels)
#     print(labels[0].item())
#     im_grid = torchvision.utils.make_grid(im, nrow=nrow)
#     logger.add_image(class_to_index[phase][labels[0].item()], im_grid, iter)

def fetch_class_name(labels, phase):
    class_list = list(map(lambda label: class_to_index[phase][label.item()], labels))
    return ' | '.join(map(str, class_list))


def get_num_of_epochs_trained():
    count = 0
    dir_path = 'trained_weights_unet_resnet_noise'
    for path in os.scandir(dir_path):
        if path.is_file():
            count += 1
    return count


def train_model(resnet, criterion, optimizer, device, tensorboard, num_epochs=3):
    num_of_epochs_trained = get_num_of_epochs_trained()
    if num_of_epochs_trained > 0:
        resume_epoch = num_of_epochs_trained - 1
        checkpoint = torch.load(f'trained_weights_unet_resnet_noise/{resume_epoch}.pth')
        resnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70], gamma=0.1)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}.")
        print('-' * 10)

        scheduler.step()

        for phase in ['train', 'validation']:
            if phase == 'train':
                resnet.train()
            else:
                resnet.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                resnet_outputs, masks, unmasks = resnet(inputs)
                loss = criterion(resnet_outputs, labels)
                # tensorboard.add_images(f'Masks for {fetch_class_name(labels, phase)}', masks, 0)
                # tensorboard.add_images(f'Masked Outputs for {fetch_class_name(labels, phase)}', masked_outputs, 0)

                # grid = torchvision.utils.make_grid(torch.cat((masks, masked_outputs, unmasks, unmasked_outputs), dim=0), nrow=len(masks))
                # tensorboard.add_image(f'{fetch_class_name(labels, phase)}', grid, i)

                if i % 4 == 3:
                    tensorboard.flush()

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(resnet_outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])

            tb.add_scalar(f"Loss during {phase}", epoch_loss, epoch)
            tb.add_scalar(f"Accuracy during {phase}", epoch_acc, epoch)

            print(f"{phase} loss: {epoch_loss.item()}, acc: {epoch_acc.item()}")
            torch.save({'epoch': epoch, 'model_state_dict': resnet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss.item()},
                       f'trained_weights_unet_resnet_noise/{epoch}.pth')
    tensorboard.close()
    return resnet


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word_list = pd.read_csv('emotion_words.csv')['words']
    # unet = UNet(3, 3).to(device)
    tb = SummaryWriter()
    uresnet = UResnetModel(len(word_list), 3, tb).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(uresnet.parameters(), lr=0.001)
    # optimizer = optim.Adam(uresnet.parameters(), lr=0.001)

    model_trained = train_model(uresnet, criterion, optimizer, device, tb, 80)
    torch.save(model_trained.state_dict(),
               'trained_weights_unet_resnet_noise/unet_resnet_model_weights_80_epoch.pth')
