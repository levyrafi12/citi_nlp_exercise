import abc
import torch
import torch.nn as nn

class Trainer(abc.ABC):
	"""
	A class implementing and abstracting various tasks of training models.
	"""
	def __init__(self, model, optimizer, n_epochs=1, 
		loss_fn=nn.BCEWithLogitsLoss(),
		# loss_fn=nn.CrossEntropyLoss(),
		print_every=10):
		"""
		Initialize the trainer.
		:param model: Instance of the model to train.
		:param loss_fn: The loss function to evaluate with.
		:param optimizer: The optimizer to train with.
		:param device: torch.device to run training on (CPU or GPU).
		:param n_epochs: how many epochs to train the model
		:param print_every:
		:param loss_fn
		"""
		self.model = model
		self.loss_fn = loss_fn
		self.n_epochs = n_epochs
		self.print_every = print_every
		self.optimizer = optimizer

	def fit(self, train_dl, valid_dl, max_batches=None, valid_max_batches=None):
		train_acc = []
		valid_acc = []
		train_losses = []
		valid_losses = []

		if max_batches == None:
			max_batches = len(train_dl)

		num_batches = min(len(train_dl), max_batches)

		print(f'Start training: num epochs {self.n_epochs} , num batches was set (possibly limited) to {num_batches}')

		for epoch in range(self.n_epochs):
			avg_loss = 0
			avg_acc = 0
			# For each batch in our training set
			for batch_ind, batch in enumerate(train_dl):
				batch = self.convert_batch(batch)
				if batch_ind >= num_batches:
					break
				input_ids, labels = batch
				batch_loss, batch_acc = self.train_on_batch(batch)

				avg_acc = (avg_acc * batch_ind + batch_acc)  / (batch_ind + 1)
				avg_loss = (avg_loss * batch_ind + batch_loss) / (batch_ind + 1)

				if (batch_ind + 1) % self.print_every == 0:
					print(f'Epoch {epoch + 1} ({batch_ind + 1}/{num_batches}) , train loss {avg_loss:.3f} , acc {avg_acc:.2f}')

			print(f'Epoch {epoch + 1} train loss {avg_loss:.3f} , acc {avg_acc:.2f}')

			train_acc.append(avg_acc)
			train_losses.append(avg_loss)

			avg_loss, avg_acc = self.evaluate(valid_dl, valid_max_batches)
			print(f'Epoch {epoch + 1} validation loss {avg_loss:.3f} , acc {avg_acc:.2f}')

			valid_losses.append(avg_loss)
			valid_acc.append(avg_acc)

		return train_losses, train_acc, valid_losses, valid_acc

	# default implementation
	def convert_batch(self, batch):
		return batch

	def train_on_batch(self, batch):
		""" 
		Run a single back forward through the model, calculates loss,
		preforms back-propagation and uses the optimizer to update weights.
		"""
		input_ids, labels = batch

		# compute logits
		# shape is (B, seq_len, num entities)
		logits = self.model(input_ids)
		avg_loss, avg_acc = self.batch_loss_compute(logits, labels)

		self.optimizer.zero_grad()
		avg_loss.backward()
		self.optimizer.step()

		return avg_loss.item(), avg_acc

	def evaluate(self, data_loader, max_batches):
		"""
		Run evaluation on the validation set. Compute average loss and accuracy.
		:param data_loader: the validation set
		:return average loss and accuracy
		"""
		max_batches = max_batches if max_batches != None \
			else len(data_loader)
		num_batches = min(max_batches, len(data_loader))

		print(f'Start evaluation: num batches was set (possibly limited) to {num_batches}')

		avg_loss = 0
		avg_acc = 0

		# For each batch in our validation set
		for batch_ind, batch in enumerate(data_loader):
			batch = self.convert_batch(batch)
			input_ids, labels = batch

			# Compute logits
			with torch.no_grad():
				logits = self.model(input_ids)

			batch_loss, batch_acc = self.batch_loss_compute(logits, labels)

			avg_loss = (avg_loss * batch_ind + batch_loss.item()) / (batch_ind + 1)
			avg_acc = (avg_acc * batch_ind + batch_acc) / (batch_ind + 1)

			if (batch_ind + 1) % self.print_every == 0:
				print(f'({batch_ind + 1}/{num_batches}) , valid loss {avg_loss:.3f} , valid acc {avg_acc:.2f}')

		return avg_loss, avg_acc

	@abc.abstractmethod
	def batch_loss_compute(self, logits, labels):
		"""
		Calculate loss and accuracy of a single batch logits
		:param logits:
		:param labels:
		:return: the value of the loss function and the accuracy
		"""
		raise NotImplementedError()
