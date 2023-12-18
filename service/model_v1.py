import tensorflow.python.keras.saving.save


def map_ids_to_data(decoded_recommendation, produtos_df):
    # Encontre o índice do produto mais próximo na representação latente
    idx_produto_recomendado = np.argmin(
        np.sum((produtos_df.iloc[:, 1:5].values - decoded_recommendation) ** 2, axis=1))

    # Obtenha o ID do produto associado
    id_produto_recomendado = produtos_df.iloc[idx_produto_recomendado, 0]

    return id_produto_recomendado

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
    from tensorflow.python.keras.saving.save import load_model
    # Gerando dados de exemplo
    # Substitua isso pelos seus próprios dados
    data = np.random.random((1000, 5))  # adicionando uma coluna para IDs de produtos

    # Criando um DataFrame fictício para representar os produtos
    produtos_df = pd.DataFrame(data, columns=["ID_produto", "tipo", "vendas", "tamanho", "preco"])

    # Parâmetros do modelo
    input_dim = 4  # número de características no seu conjunto de dados
    latent_dim = 2  # Dimensão latente, pode ser ajustada conforme necessário

    # Construindo o modelo VAE
    inputs = Input(shape=(input_dim,))
    h = Dense(2, activation='relu')(inputs)

    # Codificador
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    # Função de amostragem
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decodificador
    decoder_h = Dense(2, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Construindo o modelo completo
    vae = Model(inputs, x_decoded_mean)

    # Função de perda VAE
    xent_loss = input_dim * tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Treinando o VAE
    vae.fit(data[:, :input_dim], epochs=50, batch_size=32, shuffle=True)

    # Obtendo as representações latentes para novos dados
    encoder = Model(inputs, z_mean)  # Use z_mean para obter a representação determinística

    # Testando o modelo com alguns dados de exemplo
    input_data = np.random.random((1, input_dim))  # Substitua isso pelos seus próprios dados
    encoded_data = encoder.predict(input_data)

    # Obtendo a recomendação
    decoded_recommendation = vae.predict(input_data)

    id_produto_recomendado = map_ids_to_data(decoded_recommendation, produtos_df)

    # Agora, você pode usar o ID do produto recomendado para encontrar informações detalhadas
    produto_recomendado = produtos_df[produtos_df["ID_produto"] == id_produto_recomendado]

    print("Produto Recomendado:")
    print(produto_recomendado)


