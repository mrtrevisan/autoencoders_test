import matplotlib.pyplot as plt
import random
from keras import datasets, models
from classes.Basic import Autoencoder

# Carrega os dados para teste do FASHION MNIST
(_, _), (x_test, _) = datasets.fashion_mnist.load_data()

# Normaliza os dados: [0, 255] => [0, 1]
x_test = x_test.astype('float32') / 255.

# Carrega o modelo treinado previamente
autoencoder = models.load_model('models/basic_model.keras', custom_objects={'Autoencoder': Autoencoder})

decoded_imgs = autoencoder.call(x_test)

# Faz a "plotagem" de n imagens recontruídas, escolhidas aleatóriamente
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    rand = random.randint(0, len(decoded_imgs))

    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[rand])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[rand])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
