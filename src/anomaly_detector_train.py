import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from classes.AnomalyDetector import AnomalyDetector

# Download do dataset de ECGs, usando pandas
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# O último elemento contem os rótulos
labels = raw_data[:, -1]
# Os outros elementos são os dados
data = raw_data[:, 0:-1]

# divide os dados em dados de treino e validação 0.8 : 0.2
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# normaliza os dados
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

# faz casting para tipo de dados Float32
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# ECGs normais são rotulados com 1
# anormais com 0
# separa os dados em normais e anormais
# apenas os dados normais serão usados para treinamento
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# Inicializa o modelo
autoencoder = AnomalyDetector()
# compila o modelo com função "adam" de otimização e função loss "Mean Absolute Error"
autoencoder.compile(optimizer='adam', loss='mae')

# treina o modelo
autoencoder.fit(normal_train_data, normal_train_data,
    # numero de epocas de treinamento
    epochs=20,
    # número de amostrar por época
    batch_size=512,
    # dados de validação
    validation_data=(test_data, test_data),
    # embaralha entre cada época
    shuffle=True
)

# salva o modelo
autoencoder.save('models/anomaly_detector_model.keras')
