import random
import torch
import torch.nn as nn
import torch.optim as optim

from training import Trainer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(42)

from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import unittest

test = unittest.TestCase()

logging.set_verbosity_error()


class TagModel(nn.Module):
	"""
	A class for tag sequence classifier. It is composed of two layers:
	bert model and a feed forward layer.
	"""
	def __init__(self, in_dim, h_dim, out_dim, device, freeze):
		"""
		:param in_dim: the output dimension of the bert model
		:param h_dim: the hidden layer dimension of the FF
		:param out_dim: the output dimension (equal to the number of tag entities).
		:param freeze: If True, Bert parameters will not be fine tuned.
		"""
		super().__init__()

		self.bert = BertModel.from_pretrained('bert-base-cased').to(device)

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
		:param input_ids: the input tensor of shape (B, seq_len)
		:return the logits of all token sequence. Tensor shape (B, seq_len, num_tags)
		"""
		B = input_ids.shape[0]
		outputs = self.bert(input_ids)
		# the last hidden state
		last_hidden_state = outputs.last_hidden_state[:, :, :]
		seq_len = last_hidden_state.shape[1]
		logits = []

		for i in range(seq_len):
			# Feed input to classifier to compute logits for every input token
			logits.append(self.classifier(last_hidden_state[:, i, :]))

		return torch.hstack(logits).view((B, seq_len, -1))

class Tokenizer():
	"""
	A wrapper for bert tokenizer
	"""
	def __init__(self, device='cpu'):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
		self.device = device
		
	def preprocessing_for_bert(self, sents):
		"""
		apply bert encoding to seqeunce of words.
		:params sents: a list of words sequences.
		:return a tensor of shape (N, seq_len)
		"""
		# Create empty lists to store outputs
		input_ids = []

		for sent in sents:
			encoded_sent = self.tokenizer.encode_plus(
				text=sent,
				is_split_into_words = True,
				add_special_tokens=False,        
				max_length=64, 
				pad_to_max_length=True # Pad sentence to max length
			)
			input_ids.append(encoded_sent.get('input_ids'))

		# Convert lists to tensors
		input_ids = torch.tensor(input_ids).to(self.device)

		return input_ids

class TrainerTagging(Trainer):
	def __init__(self, model, optimizer, n_epochs):
		super().__init__(model, optimizer, n_epochs)
		"""
		:param model: the classifier
		:param optimizer:
		:param loss_fn:
		:param n_epochs:
		"""

	def batch_loss_compute(self, logits, labels):
		"""
 		Compute loss of a single batch
		:param logits: shape (B, seq_len, num entities)
		:param labels: shape (B, seq_len)
		:return average loss & accuracy
		"""

		# Get the predictions
		# shape (B, seq_len)
		preds = torch.argmax(logits, dim=2)

		avg_acc = 0
		avg_loss = 0
		seq_len = logits.shape[1]
		B = logits.shape[0]
		num_tags = logits.shape[2]
		device = logits.device

		# for every sequence in batch:
		for i in range(B):
			real_seq_len = get_real_seq_len(labels[i])
			labels_oh = torch.zeros((seq_len, num_tags)).to(device)
			for j in range(real_seq_len):
				labels_oh[j, labels[i][j]] = 1

			# compute loss
			logits_trunc = logits[i, :real_seq_len].view(\
				(-1, real_seq_len, num_tags))
			labels_oh = labels_oh[:real_seq_len, :].view(\
				(-1, real_seq_len, num_tags))
			avg_loss = (avg_loss * i + self.loss_fn(logits_trunc, labels_oh)) / (i + 1)
			avg_loss = avg_loss.squeeze()

			# compute accuracy 
			num_correct = sum([preds[i, j] == labels[i, j] \
				for j in range(real_seq_len)]).item()
			acc = num_correct / real_seq_len
			avg_acc = (avg_acc * i + acc) / (i + 1)

		return avg_loss, avg_acc

def get_real_seq_len(labels_seq):
	"""
	Get length of sequence with pads
	"""
	return sum([1 if tag_ind >= 0 else 0 for tag_ind in labels_seq])

def create_data_loader(input_ids, tag_seqs, tag_to_inds, sampler, device='cpu'):
	"""
	create data loader for train, validation and test sets (partionining into batches)
	:param input_ids: tensor of shape (N, seq_len)
	:param tag_seq: sequences of tags (NER entities)
	:param tag_to_inds: a mapping between tag and an index
	:param sampler: either RandomSampler (train data only) or SequentialSampler 
	"""
	seq_len = len(input_ids[0])

	# endcode and pad labels
	tag_seqs_encoded = []
	for tag_seq in tag_seqs:
		tag_seq = tag_seq[:seq_len]
		num_pads = seq_len - len(tag_seq)
		tag_seq_with_pads = [tag_to_inds[tag] for tag in tag_seq] + num_pads * [-1] 
		tag_seqs_encoded.append(tag_seq_with_pads)

	# Convert labels to a tensor
	y_data = torch.tensor(tag_seqs_encoded).to(device)

	# Create the DataLoader for our training/validation set
	data = TensorDataset(input_ids, y_data)
	data_loader = DataLoader(data, sampler=sampler(data), batch_size=32)

	return data_loader

def build_tag_to_inds(tag_seqs):
	"""
	build mapping between tag (NER) and index
	"""
	tag_to_inds = { 'O' : 0 }
	tag_ind = 1

	for tag_seq in tag_seqs:
		for tag in tag_seq:
			if tag_to_inds.get(tag) == None:
				tag_to_inds[tag] = tag_ind
				tag_ind += 1

	return tag_to_inds

def load_data(text_file_name):
	"""
	read lines from a text file where each line contain a token and various associated 
	tags (e.g. POS and NER). We care about NER only. Sequences are separated by empty lines. 
	:param text_file_name
	:return sentences and tag sequences
	"""
	words_in_sent = []
	data = []
	tag_seq = []

	for line in open(text_file_name, 'r'):
		line = line.strip()
		# end of a sentence/sequence
		if line == "":
			data.append((words_in_sent, tag_seq))
			words_in_sent = []
			tag_seq = []
			continue

		tokens = line.split()
		words_in_sent.append(tokens[0]) # extract the word
		tag_seq.append(tokens[-1]) # extract NER

	random.shuffle(data)
	sents, tag_seqs = zip(*data)
	
	return list(sents), list(tag_seqs)

def main():
	"""
	implement a tag classification task (NER) based on Bert model. Here we
	extract from Bert the hidden state of every token (time-step) and project
	it to the tag space (to obtain the logits or the distribution of the tags).
	"""
	print("tag (NER) sequence classification...")

	train_sents, train_tags = load_data('train.txt')
	valid_sents, valid_tags = load_data('valid.txt')
	test_sents, test_tags = load_data('test.txt')

	print(f'train {len(train_sents)} , valid {len(valid_sents)} , test {len(test_sents)}')

	tag_to_inds = build_tag_to_inds(train_tags + valid_tags)

	# Make sure you are using GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)

	# In principle we could start the training by freezing Bert parameters 
	# for a certain amount of epochs and update the FF layer parameters only, 
	# and then unfreeze Bert, namely, updating or fine tuning its parameters as well, 
	# for an additional number of epochs (normally with a smaller lr). Here, we update 
	# Bert parameters immediately, from the first epoch. 
	model = TagModel(in_dim=768, h_dim=128, out_dim=len(tag_to_inds), \
		device=device, freeze=False)

	# save few bert weights before fine tuning
	bert_weight_bef_ft = torch.clone(model.bert.encoder.\
		layer[0].attention.self.query.weight[0][0:4])

	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	tokenizer = Tokenizer(device)

	x_train = tokenizer.preprocessing_for_bert(train_sents)
	x_valid = tokenizer.preprocessing_for_bert(valid_sents)
	x_test = tokenizer.preprocessing_for_bert(test_sents)

	train_dl = create_data_loader(x_train, train_tags, tag_to_inds, RandomSampler, device)
	valid_dl = create_data_loader(x_valid, valid_tags, tag_to_inds, SequentialSampler, device)
	test_dl = create_data_loader(x_test, test_tags, tag_to_inds, SequentialSampler, device)

	trainer = TrainerTagging(model, optimizer, n_epochs=4)
	train_loss, train_acc, valid_loss, valid_acc = trainer.fit(train_dl, valid_dl)
	# print(train_loss, train_acc, valid_loss, valid_acc)

	# new values of the same weights we have kept before fine tuning
	bert_weight_aft_ft = model.bert.encoder.\
		layer[0].attention.self.query.weight[0][0:4]

	test.assertNotEqual(list(bert_weight_bef_ft), list(bert_weight_aft_ft))


if __name__ == "__main__":
	main()
