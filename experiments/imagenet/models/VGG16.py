"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.maxpool1 = nn.AvgPool2d(2)
        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.maxpool2 = nn.AvgPool2d(2)
        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.maxpool3 = nn.AvgPool2d(2)
        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:4*4*512
        self.maxpool4 = nn.AvgPool2d(2)
        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:14*14*512
        self.maxpool5 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        # self.init_epoch = args.init_epoch
        self.relu = nn.ReLU()
        self.max_active = [0] * 16

    def renew_max(self, x, y, epoch):
        # if epoch > self.init_epoch:
        x = max(x, y)
        return x

    def forward(self, x):
        # GROUP 1
        output = self.conv1_1(x)
        output = self.relu(output)
        # self.max_active[0] = self.renew_max(self.max_active[0], output.max(), epoch)
        output = self.conv1_2(output)
        output = self.relu(output)
        # self.max_active[1] = self.renew_max(self.max_active[1], output.max(), epoch)
        output = self.maxpool1(output)
        # GROUP 2
        output = self.conv2_1(output)
        output = self.relu(output)
        # self.max_active[2] = self.renew_max(self.max_active[2], output.max(), epoch)
        output = self.conv2_2(output)
        output = self.relu(output)
        # self.max_active[3] = self.renew_max(self.max_active[3], output.max(), epoch)
        output = self.maxpool2(output)
        # GROUP 3
        output = self.conv3_1(output)
        output = self.relu(output)
        # self.max_active[4] = self.renew_max(self.max_active[4], output.max(), epoch)
        output = self.conv3_2(output)
        output = self.relu(output)
        # self.max_active[5] = self.renew_max(self.max_active[5], output.max(), epoch)
        output = self.conv3_3(output)
        output = self.relu(output)
        # self.max_active[6] = self.renew_max(self.max_active[6], output.max(), epoch)
        output = self.maxpool3(output)
        # GROUP 4
        output = self.conv4_1(output)
        output = self.relu(output)
        # self.max_active[7] = self.renew_max(self.max_active[7], output.max(), epoch)
        output = self.conv4_2(output)
        output = self.relu(output)
        # self.max_active[8] = self.renew_max(self.max_active[8], output.max(), epoch)
        output = self.conv4_3(output)
        output = self.relu(output)
        # self.max_active[9] = self.renew_max(self.max_active[9], output.max(), epoch)
        output = self.maxpool4(output)
        # GROUP 5
        output = self.conv5_1(output)
        output = self.relu(output)
        # self.max_active[10] = self.renew_max(self.max_active[10], output.max(), epoch)
        output = self.conv5_2(output)
        output = self.relu(output)
        # self.max_active[11] = self.renew_max(self.max_active[11], output.max(), epoch)
        output = self.conv5_3(output)
        output = self.relu(output)
        # self.max_active[12] = self.renew_max(self.max_active[12], output.max(), epoch)
        output = self.maxpool5(output)
        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        # self.max_active[13] = self.renew_max(self.max_active[13], output.max(), epoch)
        output = self.fc2(output)
        output = self.relu(output)
        # self.max_active[14] = self.renew_max(self.max_active[14], output.max(), epoch)
        output = self.fc3(output)
        # self.max_active[15] = self.renew_max(self.max_active[15], output.max(), epoch)
        return output

    def record(self):
        return np.array([act.detach().cpu() for act in self.max_active])
        # return np.array([self.max_active)

    def load_max_active(self, mat):
        self.max_active = mat
