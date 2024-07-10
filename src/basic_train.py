from keras import losses, datasets
from classes.Basic import Autoencoder

# Carrega os dados para treinamento e teste do FASHION MNIST
(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

# Normaliza os dados: [0, 255] => [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Define os parâmetros do Autoencoder
# Formato da entrada e tamanho da camada latente
shape = x_test.shape[1:]
latent_dim = 32

# Instancia o modelo
autoencoder = Autoencoder(latent_dim, shape)
# Compila o modelo com função de otimização "adam" e função de perda Eroo Quadrático Médio
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# Treina o modelo
autoencoder.fit(
    # Usa os mesmos dados para treino e rótulo, pois é um autoencoder
    x_train, x_train,
    # Teina por 10 épocas
    epochs=10,
    # Embaralha os dados entre as épocas
    shuffle=True,
    # Dados de validação
    validation_data=(x_test, x_test)
)

# salva o modelo
autoencoder.save('models/basic_model.keras')
