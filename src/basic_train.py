from keras import losses, datasets
from classes.Basic import Autoencoder

# loads the train and test data
(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# calculate the Autoencoder parameter
shape = x_test.shape[1:]
latent_dim = 32 # the size of the bottleneck layer

# initialize the autoencoder, trains and saves it
autoencoder = Autoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(
    x_train, x_train,
    epochs=10,
    shuffle=True,
    validation_data=(x_test, x_test)
)

autoencoder.save('models/basic_model.keras')
