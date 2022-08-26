import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from pathlib import Path


 
def lowlight(image_path,root_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'	
	data_lowlight = Image.open(image_path.as_posix())

	rel_path = image_path.relative_to(root_path)
	result_path = root_path.parent / "result" / rel_path
 
	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	result_path.parent.mkdir(exist_ok=True, parents=True)
	torchvision.utils.save_image(enhanced_image, result_path.as_posix())

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
		filePath = Path(filePath)	
		
		test_list = [p for p in filePath.glob("**/custom/*jpg")]
		# test_list += [p for p in filePath.glob("*/**/*bmp")]

		for image in test_list:
			# image = image
			print(image)
			lowlight(image,filePath)

		

