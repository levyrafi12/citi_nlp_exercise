import json
from collections import defaultdict
import random
import torch
import torch.nn as nn

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
	def __init__(self, in_dim=768, h_dim=128, out_dim=5):
		"""
		:param in_dim: is the hidden state dimension of bert
		:param h_dim: hidden dimension of the feed forward layer
		:params out_dim: output dimension of shape 5
		"""
		super().__init__()
	
		# Instantiate BERT model
		self.bert = BertModel.from_pretrained('bert-base-uncased')

		# Instantiate an one-layer feed-forward classifier
		self.classifier = nn.Sequential(
 			nn.Linear(in_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, out_dim)
		)

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
	def __init__(self):
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
				max_length=64,              
				pad_to_max_length=True # Pad sentence to max length
			)
			input_ids.append(encoded_review)

		# Convert lists to tensors
		input_ids = torch.tensor(input_ids)

		return input_ids

class Trainer():
	"""
	A class of various tasks related to training
	"""
	def __init__(self, model, loss_fn=nn.CrossEntropyLoss()):
		self.model = model
		self.loss_fn = loss_fn
	
	def evaluate(self, data_loader):
		acc = []
		losses = []

		# For each batch in our data set
		for batch in data_loader:
			input_ids, labels = batch

			# Compute logits
			with torch.no_grad():
				# has the shape of (B, 5)
				logits = self.model(input_ids)

			# Compute loss
			# print(logits.shape, labels.shape)
			# shapes: logits (B, 5), lables (B,)
			loss = self.loss_fn(logits, labels)
			losses.append(loss.item())

       		 	# Get the predictions
			preds = torch.argmax(logits, dim=1)

			# Calculate the accuracy 
			num_correct = sum([preds[i] == labels[i] for i in range(len(labels))]) 
			acc.append(num_correct.item() / len(labels))
		
		return losses, acc

def create_data_loader(input_ids, labels, sampler):
	"""
	Create DataLoader for our training/validation set
	:param inputs_ids: a Tensor of shape (N, max_seq)
	:param labels: list of N review grades (between 0 to 4)
	:params sampler: either random (for training only) or sequential.
	"""
	# Convert labels to a tensor 
	y_data = torch.tensor(labels)

	# Create the DataLoader for our training/validation set
	data = TensorDataset(input_ids, y_data)
	data_loader = DataLoader(data, sampler=sampler(data), batch_size=32)
	
	return data_loader

def load_and_split_dataset(json_file_name):
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
	for i, line in enumerate(open(json_file_name, 'r')):
		# fixing the dataset to allow translating the str line into a dictionary
		line = line.replace('"verified": true,', '"verified": "true",')
		line = line.replace('"verified": false,', '"verified": "false",')
		line = eval(line)

		if line.get("reviewText") == None:
			continue

		# we only care about two fields, the review text and the overall grade (1 to 5)
		line = { 'reviewText' : line["reviewText"], 'overall': line['overall']}
		lines.append(line)

	train_ratio = 1 - test_ratio - valid_ratio
	n = len(lines)
	random.shuffle(lines)
	labels = [int(line["overall"]) - 1 for line in lines]
	
	x_train = lines[:int(n * train_ratio)]
	y_train = labels[:int(n * train_ratio)] 

	x_valid = lines[-int(n * (1 - train_ratio)): -int(n * test_ratio)]
	y_valid = labels[-int(n * (1 - train_ratio)): -int(n * test_ratio)]

	x_test = lines[-int(n * test_ratio):]
	y_test =  labels[-int(n * test_ratio):]

	print(f'train: {len(x_train)} , valid: {len(x_valid)} , test: {len(x_test)}')

	tokenizer = Tokenizer()
	
	train_inputs = tokenizer.preprocessing_for_bert(x_train)
	valid_inputs = tokenizer.preprocessing_for_bert(x_valid)
	test_inputs = tokenizer.preprocessing_for_bert(x_test)

	train_dl = create_data_loader(train_inputs, y_train, RandomSampler)
	valid_dl = create_data_loader(valid_inputs, y_valid, SequentialSampler)
	test_dl = create_data_loader(test_inputs, y_test, SequentialSampler)

	return train_dl, valid_dl, test_dl

def main():
	print("Recommendation classifer...")
	""" 
	we implemented a recommendation classifier on top of Bert. Based on a sequence of words 
	(text review) the model tries predict a single number, the recommendation grade (1-5). 
	For that task, we extract from Bert the hidden state of the special token [CLS], 
	which marks the beginning of a sequence, and we project that state to the grade space 
	(to obtain the logits or the distribution of the grades)
	"""

	train_dl, valid_dl, test_dl = load_and_split_dataset('Digital_Music.json')

	model = RecommendModel()	
	trainer = Trainer(model)
	losses, acc = trainer.evaluate(valid_dl)
	# print(losses, acc)

if __name__ == "__main__":
	main()
