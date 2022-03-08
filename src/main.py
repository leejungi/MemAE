import numpy as np
import torch
import logging
import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.metrics import roc_curve, auc

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 

from model.memae import AutoEncoderCov3DMem as memae 
from model.memae import AutoEncoderCov3D as ae 
from model.entropy_loss import *
from datasets import Dataset



if __name__ == "__main__":

	#Parameters
	device='cuda'
	normal= 1
	p=0.3
	batch=200
	lr=0.0001
	epochs= 100
	alpha= 0.0002
	model_name = "memAE"
	print(f"Model: {model_name}")
	
	aupr_list= []
	auc_list= []
	for normal in range(10):
		train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
		test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())

		train_data, train_label = train_data.data, train_data.targets
		test_data, test_label = test_data.data, test_data.targets

		n_indices = torch.where(train_label==normal)
		n_traind = train_data[n_indices]
		n_trainl = train_label[n_indices]

		ab_indices = torch.where(train_label!=normal)
		ab_traind = train_data[ab_indices]
		ab_trainl = train_label[ab_indices]

		ab_len = len(ab_traind)


#		data = torch.cat((n_traind, ab_traind[:int(ab_len*p)]))/255.0
#		label = torch.cat((n_trainl, ab_trainl[:int(ab_len*p)]))
#
#		train_len = len(data)
#		indices = np.random.permutation(train_len)
#		data=data[indices]
#		label=label[indices]
#
#		train_data = data[int(train_len/3):]
#		train_label = label[int(train_len/3):]
#		test_data = data[:int(train_len/3)]
#		test_label = label[:int(train_len/3)]
#
		data = n_traind 
		label = n_trainl 
		train_len = len(data)
		indices = np.random.permutation(train_len)
		data=data[indices]
		label=label[indices]

		train_data = data[int(train_len/3):]/255.0
		train_label = label[int(train_len/3):]

		test_data = data[:int(train_len/3)]
		test_label = label[:int(train_len/3)]
		test_len = len(test_data)
		test_data = torch.cat((test_data, ab_traind[:int(test_len*p/(1-p))]))/255.0
		test_label = torch.cat((test_label, ab_trainl[:int(test_len*p/(1-p))]))

		train_dset = Dataset(train_data,train_label,normal=normal)
		train_loader = torch.utils.data.DataLoader(dataset=train_dset, batch_size = batch, shuffle=True, drop_last=False)

		test_dset = Dataset(test_data,test_label,normal=normal)
		test_loader = torch.utils.data.DataLoader(dataset=test_dset, batch_size = batch, shuffle=True, drop_last=False)


		if model_name =="AE":
			model = ae(1).to(device)
		elif model_name =="memAE":
			model = memae(1, 500, 0.0025).to(device)
		loss_fn=nn.MSELoss().to(device)
		entropy_loss_fn=EntropyLossEncap().to(device)
		optimizer=torch.optim.Adam(model.parameters(), lr=lr)


		#Train
		for epoch in range(epochs):
			total_loss=[]
			for batch_idx, (X,Y) in enumerate(train_loader):
				X = X.to(device)
				Y = Y.to(device)

				if model_name =="AE":
					recon_x = model(X)
					loss = loss_fn(recon_x,X)
					total_loss.append(loss.item())


				elif model_name =="memAE":
					recon_res = model(X)
					recon_x = recon_res['output']
					att_w = recon_res['att']

					recon_loss = loss_fn(recon_x,X)
					entropy_loss = entropy_loss_fn(att_w)

					loss=recon_loss+entropy_loss*alpha
					total_loss.append(loss.item())

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
#		print(f"{epoch}/{epochs} Avg loss: {np.mean(total_loss)}")

		#Test
		model.eval()
		score_list = []
		label=[]
		with torch.no_grad():
			for batch_idx, (X,Y) in enumerate(test_loader):
				X = X.to(device)
				Y = Y.to(device)
				label += list(Y.to('cpu').numpy())

				recon_res = model(X)
				if model_name =="AE":
					recon_x = recon_res
				elif model_name =="memAE":
					recon_x = recon_res['output']
				X = X.view(-1,28*28)
				recon_x = recon_x.view(-1,28*28)

				score_list += torch.mean((recon_x-X)**2,1).to('cpu')
					
			score_list = np.array(score_list)
			label = np.array(label)

		precision, recall, thresholds = precision_recall_curve(label, score_list)
		aupr = average_precision_score(label, score_list)
		aupr_list.append(aupr)

		fpr, tpr, _ = roc_curve(label, score_list)
		AUC = auc(fpr,tpr)
		auc_list.append(AUC)
		print(f"Normal Class: {normal} AUPR: {aupr} AUC: {AUC}")
	aupr_list = np.array(aupr_list)
	auc_list = np.array(auc_list)
	print(f"AVG AUPR: {np.mean(aupr_list)}")
	print(f"AVG AUC: {np.mean(auc_list)}")




