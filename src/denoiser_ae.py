import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import datasets, models
from classes.Denoise import Denoise

# loads the test data
(_, _), (x_test, _) = datasets.fashion_mnist.load_data()

x_test = x_test.astype('float32') / 255.
x_test = x_test[..., tf.newaxis]

# adds noise to it
noise_factor = 0.2
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# loads the autoencoder model and uses it
autoencoder = models.load_model('models/denoise_model.keras', custom_objects={'Denoise': Denoise})
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# plots the denoised images
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
