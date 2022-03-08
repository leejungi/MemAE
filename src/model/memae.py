from __future__ import absolute_import, print_function
import torch
from torch import nn

from model.mem_module import MemModule

class AutoEncoderCov3D(nn.Module):
	def __init__(self, chnum_in):
		super(AutoEncoderCov3D, self).__init__()
		self.chnum_in = chnum_in


		self.encoder = Encoder(image_channel_size=1,
							   conv_channel_size=8
							  )

		self.decoder = Decoder(image_height=28,
							   image_width=28,
							   image_channel_size=1,
							   conv_channel_size=8
							  )
	def forward(self, x):
		f = self.encoder(x)
		output = self.decoder(f)
		return output


class AutoEncoderCov3DMem(nn.Module):
	def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
		super(AutoEncoderCov3DMem, self).__init__()
		self.chnum_in = chnum_in


		self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=8*3*3, shrink_thres =shrink_thres)
		self.encoder = Encoder(image_channel_size=1,
							   conv_channel_size=8
							  )

		self.decoder = Decoder(image_height=28,
							   image_width=28,
							   image_channel_size=1,
							   conv_channel_size=8
							  )
	def forward(self, x):
		f = self.encoder(x)
		res_mem = self.mem_rep(f)
		f = res_mem['output']
		att = res_mem['att']
		output = self.decoder(f)
		return {'output': output, 'att': att}


class Encoder(nn.Module):
	def __init__(self, image_channel_size, conv_channel_size):
		super(Encoder, self).__init__()
		self.image_channel_size = image_channel_size
		self.conv_channel_size = conv_channel_size

		self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
							   out_channels=self.conv_channel_size*4,
							   kernel_size=3,
							   stride=2,
							   padding=1,
							  )

		self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

		self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size*4,
							   out_channels=self.conv_channel_size*2,
							   kernel_size=3,
							   stride=2,
							   padding=1,
							  )

		self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

		self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size*2,
							   out_channels=self.conv_channel_size,
							   kernel_size=3,
							   stride=3,
							   padding=1,
							  )

		self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		batch, _, _, _ = x.size()
		x = x.view(batch, -1)
		return x


class Decoder(nn.Module):
	def __init__(self, image_height, image_width, image_channel_size, conv_channel_size):
		super(Decoder, self).__init__()
		self.image_height = image_height
		self.image_width = image_width
		self.image_channel_size = image_channel_size
		self.conv_channel_size = conv_channel_size

		self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
										  out_channels=self.conv_channel_size*2,
#										  kernel_size=2,
#										  stride=2,
										  kernel_size=3,
										  stride=3,
										  padding=1,
#										  output_padding=1,
										 )

		self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

		self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
										  out_channels=self.conv_channel_size*4,
#										  kernel_size=2,
										  kernel_size=3,
										  stride=2,
										  padding=1,
										  output_padding=1,
										 )

		self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

		self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
										  out_channels=self.image_channel_size,
#										  kernel_size=2,
										  kernel_size=3,
										  stride=2,
										  padding=1,
										  output_padding=1,
										 )

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = x.view(-1, self.conv_channel_size, 3,3)

		x = self.deconv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.deconv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.deconv3(x)
		return x
