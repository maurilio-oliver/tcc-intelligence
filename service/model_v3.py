import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Suponha que você tenha um DataFrame chamado 'traind_data' com colunas 'XS' e 'YS'

# Função para transformar os dados
def transform_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)
def modelo(traind_data):
    # Transformar os dados
    xs_data = transform_data(traind_data['XS'])
    ys_data = transform_data(traind_data['YS'])

    # Dividir os dados em treino e teste
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs_data, ys_data, test_size=0.2, random_state=42)

    # Construir o modelo
    model = keras.models.Sequential()
    model.add(Dense(4, activation='relu', input_dim=1))  # Ajuste input_dim para a dimensão dos seus dados
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='softmax'))  # Use 'softmax' para classificação multiclasse

    # Compilar o modelo
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Treinar o modelo
    model.fit(xs_train, ys_train, batch_size=5, epochs=10, shuffle=True)

    # Avaliar o modelo
    accuracy = model.evaluate(xs_test, ys_test)
    print(f'Acurácia do modelo: {accuracy[1]}')

if __name__ == '__main__':
    modelo({
        "XS": np.random.random_integers(low=0, high=100, size=(4,4)),
        "YS": np.random.random_integers(low=0, high=100, size=(4, 4))
    })
