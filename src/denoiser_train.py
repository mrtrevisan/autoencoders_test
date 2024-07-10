import tensorflow as tf
from keras import losses, datasets
from classes.Denoise import Denoise

# Carrega os dados para treinamento e teste do FASHION MNIST
(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

# Normaliza os dados: [0, 255] => [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Adiciona uma dimensão extra para os canais (necessário para convolução 2D)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# fator de ruído
noise_factor = 0.2
# Adiciona ruído aleatório às imagens
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Garante que os valores com ruído estão no intervalo [0, 1]
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# Inicializa o modelo
autoencoder = Denoise()
# Compila o modelo com função de otimização "adam" e função de perda Eroo Quadrático Médio
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# Treina o modelo
autoencoder.fit(
    # Usa os dados ruidosos para treino e os originais para rótulo
    x_train_noisy, x_train,
    # 10 épocas de treinamento
    epochs=10,
    # embaralha
    shuffle=True,
    # dados de validação
    validation_data=(x_test_noisy, x_test)
)

# salva o modelo
autoencoder.save('models/denoise_model.keras')
