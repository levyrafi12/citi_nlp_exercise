import json
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim

from training import Trainer

import unittest

test = unittest.TestCase()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import BertModel 
from transformers import BertTokenizer
from transformers import logging

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logging.set_verbosity_error()

random.seed(42)

valid_ratio = 0.1
test_ratio = 0.1

class RecommendModel(nn.Module):
	"""
	A class for representing the recommendation classificer.
	Model is composed of two layers: the bert model and a feed
	forward layer.
	"""
	def __init__(self, in_dim, h_dim, out_dim, device, freeze):
		"""
		:param in_dim: is the hidden state dimension of bert
		:param h_dim: hidden dimension of the feed forward layer
		:params out_dim: output dimension of shape 5 (num grades)
		"""
		super().__init__()
	
		# Instantiate BERT model
		self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)

		# Instantiate an one-layer feed-forward classifier
		self.classifier = nn.Sequential(
 			nn.Linear(in_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, out_dim)
		).to(device)

		if freeze == True:
			for param in self.bert.parameters():
				param.requires_grad = False

	def forward(self, input_ids):
		"""
		:param input_ids: an input tensor of (B, max_len)
			where each tensor's row is a sequence of encoded words.
		:return the logits tensor of shape (B, seq_len).
		"""
		outputs = self.bert(input_ids)
		# we extract the hidden state of the token [CLS] which
		# marks the beginning of the sequence.  Shape is (B, H=768)
		last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

 		# Feed input to classifier to compute logits
		# shape is (B, num_classes=5)
		logits = self.classifier(last_hidden_state_cls)
		
		return logits

class Tokenizer():
	"""
	A wrapper around bert tokenizer
	"""
	def __init__(self, device, max_len=64):
		self.device = device
		self.max_len = max_len
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	
	def preprocessing_for_bert(self, lines):
		"""
		Tokenizining the textual review into encoded words
		:param lines: list of lines where each line is kept as a dictionary
			with two fields: overall and reviewText. Here we care about the latter one.
		"""
		# Create empty lists to store outputs
		input_ids = []

		for line in lines:
			encoded_review = self.tokenizer.encode(
				text=line["reviewText"],  
				add_special_tokens=True, # Add `[CLS]` and `[SEP]`
				max_length=self.max_len,              
				pad_to_max_length=True # Pad sentence to max length
			)
			input_ids.append(encoded_review)

		# Convert lists to tensors
		input_ids = torch.tensor(input_ids).to(self.device)

		return input_ids

class TrainerRecommend(Trainer):
	"""
	A class of various tasks related to training
	"""
	def __init__(self, model, optimizer, tokenizer, lines, n_epochs, loss_fn):
		super().__init__(model, optimizer, n_epochs, loss_fn)
		"""
		:param model: the classifier
		:param optimizer
		"""
		self.lines = lines
		self.tokenizer = tokenizer

	def convert_batch(self, batch): 
		input_ids, labels = batch
		
		batch_lines = []
		for i in range(len(input_ids)):
			batch_lines.append(self.lines[input_ids[i]])
	
		batch_enc_tensor = self.tokenizer.preprocessing_for_bert(batch_lines)
		return batch_enc_tensor, labels
	
	def batch_loss_compute(self, logits, labels):
		"""
 		Compute loss of a single batch
		:param logits: shape (B, num grades)
		:param labels: shape (B,)
		:return average loss & accuracy
		"""

		# Get the predictions
		# shape (B,)
		preds = torch.argmax(logits, dim=1)

		num_classes = logits.shape[1]
		B = logits.shape[0]
		device = logits.device

		# compute loss
		loss = self.loss_fn(logits, labels)

		# compute accuracy 
		num_correct = sum([preds[i] == labels[i] for i in range(B)])

		return loss, num_correct / B

def create_data_loader(input_inds, labels, sampler, device='cpu', batch_size=32):
	"""
	Create DataLoader for our training/validation set
	:param inputs_inds: a list of line indices of size N
	:param labels: list of N review grades (between 0 to 4)
	:params sampler: either random (for training only) or sequential.
	"""
	# Convert labels to a tensor 
	y_data = torch.tensor(labels).to(device)

	# Create the DataLoader for our training/validation set
	data = TensorDataset(torch.Tensor(input_inds).to(torch.int32), y_data)
	data_loader = DataLoader(data, sampler=sampler(data), batch_size=batch_size)
	
	return data_loader

def load_and_split_dataset(json_file_name, device, max_samples=None):
	"""
	load a json file which composed of lines (records) where
	each line represents a single review on a particular product
	given by a reviwer. For our classification task, predicting the review grade 
	based on the textual review, we care about 'reviewText' and 'overall' fields only.
	After parsing and extracting the relevant information, we split the data into 
	three sets: training, validation and test. Lastly, we apply bert tokenization
	and encoding on the textual reviews. 
	:param jason_file_name
	"""
	lines = []
	line_inds = []
	line_ind = 0
	for i, line in enumerate(open(json_file_name, 'r')):
		if max_samples != None and i > max_samples:
			break
		# fixing the dataset to allow translating the str line into a dictionary
		line = line.replace('"verified": true,', '"verified": "true",')
		line = line.replace('"verified": false,', '"verified": "false",')
		line = eval(line)

		if line.get("reviewText") == None:
			continue

		# we only care about two fields, the review text and the overall grade (1 to 5)
		line = { 'reviewText' : line["reviewText"], 'overall': line['overall']}
		lines.append(line)
		line_inds.append(line_ind)
		line_ind += 1

	train_ratio = 1 - test_ratio - valid_ratio
	n = len(lines)
	random.shuffle(line_inds)
	labels = [int(lines[i]["overall"]) - 1 for i in line_inds]
	
	x_train = line_inds[:int(n * train_ratio)]
	y_train = labels[:int(n * train_ratio)] 

	x_valid = line_inds[-int(n * (1 - train_ratio)): -int(n * test_ratio)]
	y_valid = labels[-int(n * (1 - train_ratio)): -int(n * test_ratio)]

	x_test = line_inds[-int(n * test_ratio):]
	y_test =  labels[-int(n * test_ratio):]

	print(f'train: {len(x_train)} , valid: {len(x_valid)} , test: {len(x_test)}')

	
	train_dl = create_data_loader(x_train, y_train, RandomSampler, \
		device, batch_size=64)
	valid_dl = create_data_loader(x_valid, y_valid, SequentialSampler, \
		device, batch_size=64)
	test_dl = create_data_loader(x_test, y_test, SequentialSampler, \
		device, batch_size=64)

	return train_dl, valid_dl, test_dl, lines

def main():
	print("Recommendation classifer...")
	""" 
	we implemented a recommendation classifier on top of Bert. Based on a sequence of words 
	(text review) the model tries predict a single number, the recommendation grade (1-5). 
	For that task, we extract from Bert the hidden state of the special token [CLS], 
	which marks the beginning of a sequence, and we project that state to the grade space 
	(to obtain the logits or the distribution of the grades)
	"""
	
	# Make sure you are using GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)

	# http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Digital_Music.json.gz
	train_dl, valid_dl, test_dl, lines = load_and_split_dataset('Digital_Music.json', \
		device, max_samples=100000)

	# In principle we could start the training by freezing Bert parameters 
	# for a certain amount of epochs and update the FF layer parameters only, 
	# and then unfreeze Bert, namely, updating or fine tuning its parameters as well, 
	# for an additional number of epochs (normally with a smaller lr). Here, we update 
	# Bert parameters immediately, from the first epoch. 
	model = RecommendModel(in_dim=768, h_dim=128, out_dim=5, device=device, freeze=False)

	# save few bert weights before fine tuning
	bert_weight_bef_ft = torch.clone(model.bert.encoder.\
		layer[0].attention.self.query.weight[0][0:4])

	tokenizer = Tokenizer(device, max_len=64)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	trainer = TrainerRecommend(model, optimizer, tokenizer, lines, n_epochs=4, \
		loss_fn=nn.CrossEntropyLoss())
	train_losses, train_acc, valid_losses, valid_acc = trainer.fit(train_dl, valid_dl, \
		max_batches=10000)
	# print(train_losses, train_acc, valid_losses, valid_acc)

	# new values of the same weights we have kept before fine tuning
	bert_weight_aft_ft = model.bert.encoder.\
		layer[0].attention.self.query.weight[0][0:4]

	# print(f'weights before and after fine tuning {bert_weight_bef_ft} {bert_weight_aft_ft}')
	test.assertNotEqual(list(bert_weight_bef_ft), list(bert_weight_aft_ft))

if __name__ == "__main__":
	main()
