import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import datasets, models
from classes.Denoise import Denoise

# Carrega os dados para teste do FASHION MNIST
(_, _), (x_test, _) = datasets.fashion_mnist.load_data()

# Normaliza os dados: [0, 255] => [0, 1]
x_test = x_test.astype('float32') / 255.

# Adiciona uma dimensão extra para os canais (necessário para convolução 2D)
x_test = x_test[..., tf.newaxis]

# fator de ruído
noise_factor = 0.2
# Adiciona ruído aleatório às imagens
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Garante que os valores com ruído estão no intervalo [0, 1]
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# Carrega o modelo treinado previamente
autoencoder = models.load_model('models/denoise_model.keras', custom_objects={'Denoise': Denoise})
decoded_imgs = autoencoder.call(x_test_noisy)

# Faz a "plotagem" de n imagens recontruídas, escolhidas aleatóriamente
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    rand = random.randint(0, len(decoded_imgs))

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[rand]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[rand]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
