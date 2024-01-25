import torch.nn as nn
import torch
import numpy as np
from utils import create_nn
import pdb

class DDRSA(nn.Module):

	def __init__(self, input_dim, output_dim, layers_rnn,
				hidden_rnn, long_param = {}, att_param = {}, cs_param = {},
				typ = 'LSTM', optimizer = 'Adam', risks = 1, use_sigmoid=True):
		super(DDRSA, self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.optimizer = optimizer
		self.risks = risks
		self.typ = typ
		self.use_sigmoid = use_sigmoid

		# RNN model for longitudinal data
		if self.typ == 'LSTM':
			self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True)
		if self.typ == 'RNN':
			self.embedding = nn.RNN(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True,
								nonlinearity='relu')
		if self.typ == 'GRU':
			self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True)

		# Longitudinal network
		self.longitudinal = create_nn(hidden_rnn, input_dim, no_activation_last = True, **long_param)

		# Attention mechanism
		self.attention = create_nn(input_dim + hidden_rnn, 1, no_activation_last = True, **att_param)
		self.attention_soft = nn.Softmax(1) # On temporal dimension

		# # Cause specific network
		# self.cause_specific = []
		# for r in range(self.risks):
		# 	self.cause_specific.append(create_nn(input_dim + hidden_rnn, output_dim, no_activation_last = True, **cs_param))
		# self.cause_specific = nn.ModuleList(self.cause_specific)
		
		# RNN model for cause specific hazard output
		self.cause_specific_rnn = []
		self.cause_specific = []
		for r in range(self.risks):
			if self.typ == 'LSTM':
				self.cause_specific_rnn.append(nn.LSTMCell(input_dim + hidden_rnn, input_dim + hidden_rnn,
									bias=False)) # need to recursively get outputs for output_dim times to get the Q_jk at each step k 
			if self.typ == 'RNN':
				self.cause_specific_rnn.append(nn.RNN(input_dim + hidden_rnn, input_dim + hidden_rnn, layers_rnn,
									bias=False, batch_first=True,
									nonlinearity='relu'))
			if self.typ == 'GRU':
				self.cause_specific_rnn.append(nn.GRU(input_dim + hidden_rnn, input_dim + hidden_rnn, layers_rnn,
									bias=False, batch_first=True))
			self.cause_specific.append(create_nn(hidden_rnn+input_dim, 1, no_activation_last = True, **cs_param)) # generate one scaler hazard at each step
		self.cause_specific_rnn = nn.ModuleList(self.cause_specific_rnn)
		self.cause_specific = nn.ModuleList(self.cause_specific)
		self.sigmoid = nn.Sigmoid()

		# Probability
		self.soft = nn.Softmax(dim = -1) # On all observed output

	def forward(self, x):
		"""
			The forward function that is called when data is passed through DynamicDeepHit.
		"""
		if x.is_cuda:
			device = x.get_device()
		else:
			device = torch.device("cpu")

		# RNN representation - Nan values for not observed data --> padded with zero for not observerd data
		x = x.clone()
		inputmask = torch.abs(x).sum(2) == 0 #torch.isnan(x[:, :, 0])
		# x[inputmask] = 0
		hidden, _ = self.embedding(x)		
		
		# Longitudinal modelling
		longitudinal_prediction = self.longitudinal(hidden)

		# Attention using last observation to predict weight of all previously observed
		## Extract last observation (the one used for predictions)
		last_observations = ((~inputmask).sum(axis = 1) - 1)
		last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
		index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

		last = index == last_observations_idx
		x_last = x[last]

		## Concatenate all previous with new to measure attention
		concatenation = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1)

		## Compute attention and normalize
		attention = self.attention(concatenation).squeeze(-1)
		attention[index >= last_observations_idx] = -1e10 # Want soft max to be zero as values not observed
		attention[last_observations > 0] = self.attention_soft(attention[last_observations > 0]) # Weight previous observation
		attention[last_observations == 0] = 0 # No context for only one observation

		# Risk networks
		# The original paper is not clear on how the last observation is
		# combined with the temporal sum, other code was concatenating them
		# outcomes = []
		attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
		hidden_attentive = torch.sum(attention * hidden, axis = 1)
		hidden_attentive = torch.cat([hidden_attentive, x_last], 1) # (N, input_dim + hidden_rnn)
		
		outcomes = torch.zeros((x.shape[0], self.risks, self.output_dim))
		if self.typ == 'LSTM':
			hidden_state = (hidden_attentive.detach(), hidden_attentive.detach())
		else:
			hidden_state = hidden_attentive.detach()
		
		outcomes = []
		# for each of the output time dimension
		for k in range(self.output_dim): 
			 # for each of the risk
			for r, (cs_rnn, cs_nn) in enumerate(zip(self.cause_specific_rnn, self.cause_specific)):
				hidden_state = cs_rnn(hidden_attentive, hidden_state)
				if self.use_sigmoid:
					out = self.sigmoid(cs_nn(hidden_state[0]))
				else:
					out = cs_nn(hidden_state[0])
				outcomes.append(out)


		# for cs_nn in self.cause_specific:
		# 	outcomes.append(cs_nn(hidden_attentive))

		# Soft max for probability distribution
		outcomes = torch.cat(outcomes, dim = 1)
		outcomes = self.soft(outcomes)
		# outcomes = self.soft(outcomes.reshape(-1, self.risks * self.output_dim)).reshape(-1, self.risks, self.output_dim)

		outcomes = [outcomes[:, i * self.output_dim : (i+1) * self.output_dim] for i in range(self.risks)]
		return longitudinal_prediction, outcomes
