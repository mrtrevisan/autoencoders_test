import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from classes.AnomalyDetector import AnomalyDetector
import random

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
test_data = tf.cast(test_data, tf.float32)

# ECGs normais são rotulados com 1
# anormais com 0
# separa os dados em normais e anormais
test_labels = test_labels.astype(bool)

normal_test_data = test_data[test_labels]

anomalous_test_data = test_data[~test_labels]

# Carrega o modelo treinado previamente
autoencoder = models.load_model('models/anomaly_detector_model.keras', custom_objects={'AnomalyDetector': AnomalyDetector})

normal_decoded_data = autoencoder.call(normal_test_data)

anomalous_decoded_data = autoencoder.call(anomalous_test_data)

plt.figure(figsize=(12, 5)) 
rand = random.randint(0, len(normal_decoded_data))

# Plotagem dos dados reconstruídos
plt.subplot(1, 2, 1) 
plt.plot(normal_test_data[rand], 'b')
plt.plot(normal_decoded_data[rand], 'r')
plt.fill_between(np.arange(140), normal_decoded_data[rand], normal_test_data[rand], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title('Normal Test Data') 

plt.subplot(1, 2, 2)
plt.plot(anomalous_test_data[rand], 'b')
plt.plot(anomalous_decoded_data[rand], 'r')
plt.fill_between(np.arange(140), anomalous_decoded_data[rand], anomalous_test_data[rand], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title('Anomalous Test Data')

plt.tight_layout()
plt.show()
