
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
import time
import math
from torch.nn import init
import copy
import cv2
from skimage.measure import compare_ssim as ssim

from argparse import ArgumentParser
from utility_for_opinenet import *
from model import *
from thop import profile
from ptflops import get_model_complexity_info

seed = 6
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子

parser = ArgumentParser(description='OPINE-Net-plus')

# parser.add_argument('--epoch_num', type=int, default=98, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=6, help='phase number of OPINE-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()


# epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089


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



# Define DR-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        #########
        self.beta_step = nn.Parameter(torch.Tensor([0.1]))

        self.net = Uformer(img_size=264,in_chans=1,win_size=6,token_mlp='ffn').cuda()



    def forward(self, x_input, PhiWeight, PhiTWeight, PhiTb):
        x = x_input - self.lambda_step * PhiTPhi_fun(x_input, PhiWeight, PhiTWeight)
        x = x + self.lambda_step * PhiTb
        q = x

        z = self.net(2 * q - x_input)
        out = x_input + self.beta_step * (z - q)

        return out


# Define OPINE-Net-plus
class OPINENetplus(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(OPINENetplus, self).__init__()

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
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb)

        x_final = x

        return [x_final, Phi]


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)


model = OPINENetplus(layer_num, n_input)
model = nn.DataParallel(model)
model = model.to(device)

# flops, params = get_model_complexity_info(model, (1,264,264), as_strings=True, print_per_layer_stat=True)  #(3, 320, 320)
# print("%s |%s |%s" % ('DR-TransNet', flops, params))


model_dir = "./%s/CS_OPINE_Net_plus_layer_%d_group_%d_ratio_%d" % (args.model_dir, layer_num, group_num, cs_ratio)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params.pkl' % (model_dir)))

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
TIME_ALL = np.zeros([1, ImgNum], dtype=np.float32)



print('\n')
print("CS Sampling and Reconstruction by OPINE-Net plus Start")
print('\n')
with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]  #(np(256,256))

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)

        Img_output = Ipad.reshape(1, 1, row_new, col_new)/255.0

        start = time.time()

        batch_x = torch.from_numpy(Img_output)  #(1,1,264,264)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)


        [x_output, Phi] = model(batch_x)

        # flops, params = profile(model, inputs=(batch_x.cuda(),), verbose=True)
        # print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))

        end = time.time()
        rec_TIME = end - start

        Prediction_value = x_output.cpu().data.numpy().squeeze()  #(264,264)

        X_rec = np.clip(Prediction_value[:row,:col], 0, 1).astype(np.float64)  #(256,256)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, rec_TIME, rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        resultName = imgName.replace(args.data_dir, args.result_dir)
        cv2.imwrite("%s_OPINE_Net_plus_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio,  rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM
        TIME_ALL[0, img_no] = rec_TIME

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f/%.4f \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(TIME_ALL))
print(output_data)


print("CS Sampling and Reconstruction by OPINE-Net plus End")
