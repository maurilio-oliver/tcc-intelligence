from sklearn.cluster import KMeans
import numpy as np

# Supondo que historical_data seja uma matriz de características pré-processada
# onde cada linha representa um usuário e cada coluna representa um produto,
# com valores binários indicando interação (1 se clicou, 0 se não clicou).

# Número de clusters desejado
num_clusters = 5

# Inicialize e ajuste o modelo K-Means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(historical_data)

# Determine o cluster do usuário específico (user_id) com base em seu histórico de interações
user_cluster = kmeans.predict(user_history.reshape(1, -1))

# Recomende produtos do mesmo cluster ou clusters semelhantes
recommended_products = []
for cluster_label in similar_clusters(user_cluster):
    products_in_cluster = get_products_in_cluster(cluster_label)
    recommended_products.extend(products_in_cluster)


# Exemplo de função para encontrar clusters semelhantes (pode ser personalizada)
def similar_clusters(user_cluster):
    # Lógica para encontrar clusters semelhantes ao do usuário
    # Por exemplo, clusters com centróides próximos ao centróide do cluster do usuário
    return [user_cluster]


# Exemplo de função para obter produtos em um cluster (pode ser personalizada)
def get_products_in_cluster(cluster_label):
    # Lógica para obter produtos no cluster especificado
    # Por exemplo, consultando o banco de dados com produtos associados a esse cluster
    return []


from sklearn.cluster import KMeans
import numpy as np

# Supondo que 'data' seja uma lista de dicionários com informações dos produtos visitados pelos usuários
# Cada dicionário possui as chaves 'id', 'name', 'size' e 'category'

# Converta os dados para um formato adequado para o K-Means
X = []
for product in data:
    # Crie um vetor de características para cada produto
    # Por exemplo, [id, size] ou [id, size, category], dependendo das características que deseja considerar
    # Certifique-se de converter dados categóricos (como 'category') em valores numéricos se necessário
    feature_vector = [product['id'], product['size']]
    X.append(feature_vector)

# Converta X para um array numpy
X = np.array(X)

# Número de clusters desejado
num_clusters = 5

# Inicialize e ajuste o modelo K-Means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)


# Exemplo de função para recomendar produtos para um usuário específico
def recommend_products(user_history):
    # user_history é uma lista de IDs dos produtos que o usuário visitou

    # Transforme o histórico do usuário em um vetor de características
    user_vector = []
    for product_id in user_history:
        # Crie um vetor de características com base nos produtos visitados pelo usuário
        # Por exemplo, [product_id, product_size] ou [product_id, product_size, product_category]
        # Certifique-se de considerar apenas os produtos que estão no histórico do usuário
        # e converter dados categóricos em valores numéricos se necessário
        product = next((p for p in data if p['id'] == product_id), None)
        if product:
            feature_vector = [product['id'], product['size']]
            user_vector.append(feature_vector)

    # Se o usuário tiver histórico suficiente para formar um vetor de características
    if user_vector:
        user_vector = np.array(user_vector)
        # Determine o cluster do usuário com base em seus produtos visitados
        user_cluster = kmeans.predict(user_vector)

        # Recomende produtos do mesmo cluster ou clusters semelhantes
        recommended_products = []
        for cluster_label in user_cluster:
            products_in_cluster = [p for p in data if kmeans.predict([[p['id'], p['size']]]) == cluster_label]
            recommended_products.extend(products_in_cluster)

        return recommended_products
    else:
        return []


# Exemplo de uso
user_history = [1, 3, 5]  # IDs dos produtos que o usuário visitou
recommended_products = recommend_products(user_history)
print(recommended_products)