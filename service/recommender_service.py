import json

import pandas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import datetime



def reneme_columns(data: []):
    placeholder = {}
    if data is not None and len(data) > 0:
        for index, i in enumerate(data[0]):
            placeholder[index] = i
            if index == 'category':
                for sub_index, sub_i in enumerate(i):
                    placeholder[sub_index] = sub_i
    return placeholder


def transform(date: []):
    transformList = []

    for i in date:
        placeholder = {}
        for keys in i.keys():
            if keys == 'category':
                for sub_keys in i[keys]:
                    placeholder[sub_keys] = i[keys][sub_keys]
            else:
                placeholder[keys] = i[keys]
        transformList.append(placeholder)
    return transformList


def get_frequenci(data: pandas.DataFrame):
    rec = data['Cluster'].value_counts().idxmax()
    return data[data['Cluster'] == rec]


def get_recommender(data=[], user_preference={}):
    data.append(user_preference)
    df = pd.DataFrame(transform(data))
    df.rename(columns=reneme_columns(data))
    transform_data = df.set_index('id')
    scaler = StandardScaler()
    scale_data = scaler.fit_transform(df[['price', 'quality', 'clicks', 'sellers', 'type']])
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scale_data)
    transform_data['PCA1'] = pca_data[:, 0]
    transform_data['PCA2'] = pca_data[:, 1]

    kmeans = KMeans(n_clusters=transform_data['type'].nunique())

    transform_data['Cluster'] = kmeans.fit_predict(pca_data)

    filter_data = transform_data[transform_data['Cluster'] == transform_data.loc[user_preference['id']]['Cluster']]
    recommender = []
    for index, i in df.iterrows():
        if index in filter_data.index:
            if json.loads(i.to_json()).get('name') is not None:
                recommender.append(json.loads(i.to_json()))

    return recommender

def debgger(transform_data):
    plt.scatter(transform_data['PCA1'], transform_data['PCA2'], c=transform_data['Cluster'], cmap='viridis')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Agrupamento de Roupas por Cor, Tamanho e Tipo')
    plt.show()

if __name__ == '__main__':
    pass


