import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Kmeans:
    def __init__(self):
        self.model = KMeans(n_clusters=50, random_state=42)
        self.model_flag = False
        self.model_name = ''

    def fit_save_model(self, data, modelname='KMeans'):
        self.model.fit(data)
        self.model_flag = True
        self.model_name = f'{modelname}-{data.shape[0]}'
        with open(f'models/{self.model_name}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def pred(self, df, data):
        pred = self.model.predict(data)
        df = pd.concat([df['id'].reset_index(drop=True), pd.Series(pred)],
                       axis=1, ignore_index=False)
        df.rename(columns={0: "class"}, inplace=True)
        df[df['class'] == 0] = 50
        assert df['class'].min() > 0
        assert df['class'].max() <= 50
        return df


class Preprocessor:
    def __init__(self):
        self.vectorizer_meta = TfidfVectorizer()
        self.vectorizer_vector = TfidfVectorizer()

    def fit_save_vectorizers(self, df):
        try:
            meta1 = df['meta1']
            vector = df['vector']
        except ValueError:
            raise ValueError
        self.vectorizer_meta.fit(meta1)
        self.vectorizer_vector.fit(vector)
        with open(f'models/vectorizers/vector-meta-{df.shape[0]}.pkl', 'wb') as f:
            pickle.dump(self.vectorizer_meta, f)
        with open(f'models/vectorizers/vector-vector{df.shape[0]}.pkl', 'wb') as f:
            pickle.dump(self.vectorizer_vector, f)

    @staticmethod
    def transform_data(df):
        try:
            meta1 = df['meta1']
            vector = df['vector']
        except ValueError:
            raise ValueError
        with open(f'models/vectorizers/vector-meta-{df.shape[0]}.pkl', 'rb') as f:
            vectorizer_meta = pickle.load(f)
        with open(f'models/vectorizers/vector-vector{df.shape[0]}.pkl', 'rb') as f:
            vectorizer_vector = pickle.load(f)
        x_meta = vectorizer_meta.transform(meta1)
        x_vector = vectorizer_vector.transform(vector)
        data = hstack((x_meta, x_vector))
        return data
