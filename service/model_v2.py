import keras
import numpy as np
import tensorflow as tf
from keras.src.applications.densenet import layers
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model
from  tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def save_model(model:Model):

    save_model(model=model)
    pass

def loading_model():
    model = load_model("../resource/model.keras")
    return model
    pass

def training_model(data:[], model:Model):
    model.fit(np.array(data), epochs=50, batch_size=32, shuffle=True)

    pass

def test_model():
    model = Model()




    pass

def generate_model(input_dim):
    dense1 = Dense(units=2, activation='relu')(input_dim)
    pass

def get_recomendation():
    pass
def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * 1

def actual_model(

):
    # Gerando dados de exemplo
    # Substitua isso pelos seus próprios dados
    data = np.random.random((1000, 4))

    # Parâmetros do modelo
    input_dim = 4  # número de características no seu conjunto de dados
    latent_dim = 2  # Dimensão latente, pode ser ajustada conforme necessário

    # Construindo o modelo VAE
    inputs = Input(shape=(input_dim,))
    h = Dense(2, activation='relu')(inputs)

    # Codificador
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decodificador
    decoder_h = Dense(2, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Construindo o modelo completo
    model = Model(inputs, x_decoded_mean)

    # Função de perda VAE
    xent_loss = input_dim * tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    # Treinando o VAE
    model.fit(data, epochs=50, batch_size=32, shuffle=True)

    # Obtendo as representações latentes para novos dados
    encoder = Model(inputs, z_mean)

    input_data = np.random.random((1, input_dim))
    encoded_data = encoder.predict(input_data)

    # Obtendo a recomendação
    decoded_recommendation = model.predict(input_data)

    print("Dados codificados:")
    print(encoded_data)
    print("Recomendação:")
    print(decoded_recommendation)

    model.save("../resource/model")



if __name__ == '__main__':
    actual_model()
    model = loading_model()
    pass
