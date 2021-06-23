import random
import torch
import torch.nn as nn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(42)

from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logging.set_verbosity_error()


class TagModel(nn.Module):
	"""
	A class for tag sequence classifier. It is composed of two layers:
	bert model and a feed forward layer.
	"""
	def __init__(self, in_dim, h_dim, out_dim):
		"""
		:param in_dim: the output dimension of the bert model
		:param h_dim: the hidden layer dimension of the FF
		:param out_dim: the output dimension (equal to the number of tag entities).
		"""
		super().__init__()

		self.bert = BertModel.from_pretrained('bert-base-cased')

		# Instantiate an one-layer feed-forward classifier
		self.classifier = nn.Sequential(
			nn.Linear(in_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, out_dim)
		)

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
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
		
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
		input_ids = torch.tensor(input_ids)

		return input_ids

class Trainer():
	"""
	a class which contains various tasks related to training.
	"""
	def __init__(self, model, loss_fn=nn.CrossEntropyLoss()):
		"""
		:param model: the classifier
		:param loss_fn
		"""
		self.model = model
		self.loss_fn = loss_fn

	def evaluate(self, data_loader):
		"""
		Apply batch predictions
		:param data_loader: the validation set
		:return the loss and acc per batch 
		"""
		acc = []
		losses = []

		print(f'evaluate... num batches {len(data_loader)}')

		# For each batch in our validation set
		for batch in data_loader:
			input_ids, labels = batch

			# Compute logits
			with torch.no_grad():
				logits = self.model(input_ids)

			# print(logits.shape, labels.shape)

			# Get the predictions
			preds = torch.argmax(logits, dim=2)

			batch_loss = 0.0
			batch_num_correct = 0.0
			seq_len = logits.shape[1]
			B = logits.shape[0]

			# for every sequence in batch, compute the loss/acc for every token
			for i in range(B):
				loss = 0.0
				num_correct = 0.0
				real_seq_len = sum([1 if tag_ind >= 0 else 0 for tag_ind in labels[i]])
				num_correct = sum([preds[i, j] == labels[i, j] for j in range(real_seq_len)])
				num_correct = num_correct / real_seq_len
				batch_num_correct += num_correct

				# Compute loss
				loss += self.loss_fn(logits[i, :real_seq_len, :], labels[i, :real_seq_len]) 

				loss = loss / real_seq_len
				batch_loss += loss

			losses.append(loss.item() / B)
			acc.append(batch_num_correct.item() / B)

			
		return losses, acc

def create_data_loader(input_ids, tag_seqs, tag_to_inds, sampler):
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
	y_data = torch.tensor(tag_seqs_encoded)

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
	tokenizer = Tokenizer()

	x_train = tokenizer.preprocessing_for_bert(train_sents)
	x_valid = tokenizer.preprocessing_for_bert(valid_sents)
	x_test = tokenizer.preprocessing_for_bert(test_sents)

	train_dl = create_data_loader(x_train, train_tags, tag_to_inds, RandomSampler)
	valid_dl = create_data_loader(x_valid, valid_tags, tag_to_inds, SequentialSampler)
	test_dl = create_data_loader(x_test, test_tags, tag_to_inds, SequentialSampler)

	tag_model = TagModel(in_dim=768, h_dim=128, out_dim=len(tag_to_inds))
	trainer = Trainer(tag_model)
	loss, acc = trainer.evaluate(valid_dl)
	
	# print(loss, acc)

if __name__ == "__main__":
	main()
