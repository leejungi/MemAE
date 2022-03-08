import numpy as np
import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, label, normal=0):
		super(Dataset, self).__init__()

		self.data = data
		self.label = label
		self.normal = normal


		self.label = np.where(self.label == normal, 0, 1)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data = self.data[index]
		data = data.unsqueeze(0) 
		label = self.label[index]
		return data, label 
