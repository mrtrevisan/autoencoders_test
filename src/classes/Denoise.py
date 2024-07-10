import tensorflow as tf
from keras import layers, models

class Denoise(models.Model):
    def __init__(self):
        super(Denoise, self).__init__()

        self.encoder = tf.keras.Sequential([
            # Camada de entrada com formato 28 x 28 x 1
            layers.Input(shape=(28, 28, 1)),
            # Camada Convolucional com 16 filtros, kernel 3x3, ativação ReLU,
            # padding "same" (output com mesmo tamanho do input), e stride (passo) 2
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),  
            # Camada Convolucional com 8 filtros, kernel 3x3, ativação ReLU,
            # padding "same" (output com mesmo tamanho do input), e stride (passo) 2
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = tf.keras.Sequential([
            # Camada Convolucional Transposta com 8 filtros, kernel 3x3, ativação ReLU,
            # padding "same" (output com mesmo tamanho do input), e stride (passo) 2
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            # Camada Convolucional Transposta com 16 filtros, kernel 3x3, ativação ReLU, 
            # padding "same" (output com mesmo tamanho do input), e stride (passo) 2
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            # Camada Convolucional com 1 filtro, kernel 3x3, ativação sigmoid, e padding "same"
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    # Função que chama o processo de codif. decodif. 
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # Funções de serialização
    def get_config(self):
        config = super(Denoise, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls()
    