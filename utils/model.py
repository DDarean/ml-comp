import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Kmeans:
    def __init__(self):
        self.vectorizer_meta = TfidfVectorizer()
        self.vectorizer_vector = TfidfVectorizer()
        self.model = KMeans(n_clusters=50, random_state=42)
        self.model_flag = False

    def fit_save_model(self, df, modelname='KMeans'):
        try:
            meta1 = df['meta1']
            vector = df['vector']
        except ValueError:
            raise ValueError
        x_meta = self.vectorizer_meta.fit_transform(meta1)
        x_vector = self.vectorizer_vector.fit_transform(vector)
        data = hstack((x_meta, x_vector))
        self.model.fit(data)
        self.model_flag = True
        self.save_model(self.model, f'models/{modelname}-{df.shape[0]}.pkl')
        self.save_model(self.vectorizer_meta,
                        f'models/vectorizers/{modelname}-{df.shape[0]}-meta.pkl')
        self.save_model(self.vectorizer_vector,
                        f'models/vectorizers/{modelname}-{df.shape[0]}-vctr.pkl')

    def save_model(self, obj, filename):
        if self.model_flag:
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)

    def load_model(self, model, vectorizer_meta, vectorizer_vector):
        with open(model, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_meta, 'rb') as f:
            self.vectorizer_meta = pickle.load(f)
        with open(vectorizer_vector, 'rb') as f:
            self.vectorizer_vector = pickle.load(f)

    def preprocess_data_for_predict(self, df, model, vectorizer_meta,
                                    vectorizer_vector):
        self.load_model(model, vectorizer_meta, vectorizer_vector)
        try:
            x1 = self.vectorizer_meta.transform(df['meta1'])
            x2 = self.vectorizer_vector.transform(df['vector'])
        except ValueError:
            raise ValueError
        vectorized_data = hstack((x1, x2))
        return vectorized_data

    def pred(self, df, model, vectorizer_meta, vectorizer_vector):
        vectorized_data = self.preprocess_data_for_predict(df, model,
                                                           vectorizer_meta,
                                                           vectorizer_vector)
        pred = self.model.predict(vectorized_data)
        df = pd.concat([df['id'].reset_index(drop=True), pd.Series(pred)],
                       axis=1, ignore_index=False)
        df.rename(columns={0: "class"}, inplace=True)
        df[df['class'] == 0] = 50
        assert df['class'].min() > 0
        assert df['class'].max() <= 50
        return df
