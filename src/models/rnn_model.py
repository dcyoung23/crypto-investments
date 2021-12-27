import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from ..utils.helpers import *

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		self.model = load_model(filepath)

	def build_model(self, model_configs):

		for layer in model_configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		opt = Adam(**model_configs['model']['optimizer'])
		self.model.compile(loss=model_configs['model']['loss'], optimizer=opt)

	def train(self, X, y, train_configs):
		if train_configs['model_checkpoint']:
			callbacks = [EarlyStopping(monitor='val_loss', patience=train_configs['patience']),
			ModelCheckpoint(filepath=train_configs['save_fname'], monitor='val_loss', save_best_only=True)
		]
		else:
			callbacks = [EarlyStopping(monitor='val_loss', patience=train_configs['patience'])]

		self.model.fit(
			X,
			y,
			epochs=train_configs['epochs'],
			batch_size=train_configs['batch_size'],
            verbose=train_configs['verbose'],
			callbacks=callbacks,
            validation_split=train_configs['validation_split']
		)
		self.model.save(train_configs['save_fname'])

	def predict_point_by_point(self, data):
		# Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted


	def predict_sequences_once(self, data, steps_ahead):
		# Predict sequences at once but then limit prediction sequences at each step only
		# This will match the regressive method below
		predicted = self.model.predict(data)
		predicted = predicted[0::steps_ahead]
		# To match autoregressive sequences convert to a list of lists
		prediction_seqs = [list(p) for p in predicted]
		return prediction_seqs


	def predict_sequences_multiple(self, data, window_size, prediction_len):
		# Predict for each sequence before shifting prediction forward
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs


	def predict_sequence_full(self, data, window_size):
		# Shift the window by 1 new prediction each time, re-run predictions on new window
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted

	
    
