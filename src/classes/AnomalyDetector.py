import tensorflow as tf
from keras import layers, models

class AnomalyDetector(models.Model):
	def __init__(self):
		super(AnomalyDetector, self).__init__()
		self.encoder = tf.keras.Sequential([
			# Camada densa com 32 neurônios, ativação ReLU
			layers.Dense(32, activation="relu"),
			# Camada densa com 16 neurônios, ativação ReLU
			layers.Dense(16, activation="relu"),
			# Camada densa com 8 neurônios, ativação ReLU
			# Camada Latente ou bottleneck
			layers.Dense(8, activation="relu")
		])

		self.decoder = tf.keras.Sequential([
			# Camada densa com 16 neurônios, ativação ReLU
			layers.Dense(16, activation="relu"),
			# Camada densa com 32 neurônios, ativação ReLU
			layers.Dense(32, activation="relu"),
			# Camada densa com 140 neurônios, ativação sigmoid
			layers.Dense(140, activation="sigmoid")
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
	
	# Funções de serialização
	def get_config(self):
		config = super(AnomalyDetector, self).get_config()
		return config

	@classmethod
	def from_config(cls, config):
		return cls()
    