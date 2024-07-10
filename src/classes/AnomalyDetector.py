import tensorflow as tf
from keras import layers, models

class AnomalyDetector(models.Model):
	def __init__(self):
		super(AnomalyDetector, self).__init__()
		self.encoder = tf.keras.Sequential([
			layers.Dense(32, activation="relu"),
			layers.Dense(16, activation="relu"),
			layers.Dense(8, activation="relu")
		])

		self.decoder = tf.keras.Sequential([
			layers.Dense(16, activation="relu"),
			layers.Dense(32, activation="relu"),
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
    