import tensorflow as tf
from keras import losses, datasets
from classes.Denoise import Denoise

# loads the train and test data
(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

# normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# adds noise to the images
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# initialize the Denoise class, compile the model and train it
autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# save the model to a .keras file
autoencoder.save('models/denoise_model.keras')
