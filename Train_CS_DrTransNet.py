import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from utility_for_DrTransNet import *
from model import *

parser = ArgumentParser(description='DrTransNet')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=110, help='epoch number of end training')
parser.add_argument('--image_size', default=256, type=int, help='image size')
parser.add_argument('--layer_num', type=int, default=6, help='phase number of DrTransNet')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--train_data', default='./data/WED4744', type=str, help='path of train data')
# parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--save_interval', type=int, default=1, help='interval of saving model')


args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
image_size = args.image_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089




train_files = datagenerator(data_dir=args.train_data)



class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply



# Define DrTransNet Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        #########
        self.beta_step = nn.Parameter(torch.Tensor([0.1]))

        self.net = TransNet(img_size=264,in_chans=1,win_size=6,token_mlp='ffn').cuda()   #token_mlp='ffn','leff'


    def forward(self, x_input, PhiWeight, PhiTWeight, PhiTb):
        x = x_input - self.lambda_step * PhiTPhi_fun(x_input, PhiWeight, PhiTWeight)
        x = x + self.lambda_step * PhiTb
        q = x

        z = self.net(2 * q - x_input)
        out = x_input + self.beta_step * (z - q)

        return out


# Define DrTransNet
class DrTransNet(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(DrTransNet, self).__init__()

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x):  #(1,1,264,264)

        # Sampling-subnet
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_  #(272,1089)
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)  #(272,1,33,33)
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements (1,272,8,8)  264/33=8

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)  #(1089,272,1,1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None) #(1,1089,1,1)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb) #(1,1,264,264)
        x = PhiTb    # Conduct initialization (1,1,264,264)

        # Recovery-subnet
        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb)

        x_final = x

        return [x_final, Phi]


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)


model = DrTransNet(layer_num, n_input)
model = nn.DataParallel(model)
model = model.to(device)


print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_DrTransNet_layer_%d_group_%d_ratio_%d" % (args.model_dir, layer_num, group_num, cs_ratio)

log_file_name = "./%s/Log_CS_DrTransNet_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


Eye_I = torch.eye(n_input).to(device)


# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    for i in range(len(train_files)):

        img = train_files[i]
        Iorg_y = gray_crop_img(img, image_size)
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        Img_output = Ipad.reshape(1, 1, row_new, col_new) / 255.0  #(1,1,264,264)
        batch_x = torch.from_numpy(Img_output)  #(1,1,264,264)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device) #(1,1,264,264)

        [x_output, Phi] = model(batch_x)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_orth = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1))-Eye_I, 2))

        gamma = torch.Tensor([0.01]).to(device)
        mu = torch.Tensor([0.01]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(mu, loss_orth)

        if i % 1 == 0:
            print("Epoch:[%2d] [%4d/%4d], loss = %.6f"%(epoch_i, i, len(train_files), loss_all.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

    output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f, Orth Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_orth.item())
    print(output_data)

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
