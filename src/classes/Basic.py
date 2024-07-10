import tensorflow as tf
from keras import layers, models

class Autoencoder(models.Model):
    def __init__(self, latent_dim, shape, name=None):
        super(Autoencoder, self).__init__(name=name)
        # dimensão da camada latente (bottleneck)
        self.latent_dim = latent_dim
        # formato da entrada/saída
        self.shape = shape

        self.encoder = tf.keras.Sequential([
            # transforma a matriz de entrada em um vetor
            # * 28 x 28 x 1 => 784
            layers.Flatten(),
            # Camada totalmente conectada de tam 128, ativação ReLU
            layers.Dense(128, activation='relu'),  
            # Camada totalmente conectada de tam 64, ativação ReLU
            layers.Dense(64, activation='relu'),
            # * Camada latente de tamanho predefinido (bottleneck)
            layers.Dense(latent_dim, activation='relu'),  
        ])

        self.decoder = tf.keras.Sequential([
            # Camada totalmente conectada de tam 64, ativação ReLU
            layers.Dense(64, activation='relu'),
            # Camada totalmente conectada de tam 128, ativação ReLU
            layers.Dense(128, activation='relu'),
            # * Camada totalmente conectada do mesmo tamanho do input, ativação sigmoid
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'), 
            # * Recupera o formato da imagem original 
            layers.Reshape(shape)
        ])

    # Função que chama o processo de codif. decodif.
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # Função de serialização
    def get_config(self):
        return {
            'latent_dim' : self.latent_dim,
            'shape' : self.shape,
            'name' : self.name
        }
    