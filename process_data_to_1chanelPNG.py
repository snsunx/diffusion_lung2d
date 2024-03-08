import torch
import matplotlib.pyplot as plt 
from img_unet2d_ms import aera1d_conv
from PIL import Image
import numpy as np
from torchvision import transforms
import os 

output = []
output_directory_path = '/home/peter/data/2dmap'
files = os.listdir(output_directory_path)
for file in files:
    output.append(file)    

data = [['/home/peter/data/2dmap/' + output_name] 
        for output_name in output]

directory = '/home/ismail/diffusion_lung_2d_1chanel/datasets/lung_8_256/'
for path in data:
    target = torch.load(path[0])[:32, :]
    target = aera1d_conv(target, 4, stride=4)
    target = target.reshape(target.shape[0], target.shape[1], 1)
    uni_chanel_tensor = torch.transpose(target,0,2).transpose(1,2)
    to_pil = transforms.ToPILImage()
    rgb_image = to_pil(uni_chanel_tensor)
    name = path[0].split('/')[-1].split('.')[0]
    rgb_image.save(directory + f'{name}.png')