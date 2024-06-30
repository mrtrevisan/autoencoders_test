import tensorflow as tf
from keras import layers, models

class Autoencoder(models.Model):
    def __init__(self, latent_dim, shape, name=None):
        super(Autoencoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.shape = shape

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        return {
            'latent_dim' : self.latent_dim,
            'shape' : self.shape,
            'name' : self.name
        }
    