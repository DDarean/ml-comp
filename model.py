import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_model(df):
    try:
        meta1 = df['meta1']
        vector = df['vector']
    except ValueError:
        raise ValueError

    vectorizer_meta1 = TfidfVectorizer()
    vectorizer_vector = TfidfVectorizer()

    x_meta1 = vectorizer_meta1.fit_transform(meta1)
    x_vector = vectorizer_vector.fit_transform(vector)

    data = hstack((x_meta1, x_vector))

    model = KMeans(n_clusters=50, random_state=42)
    model.fit(data)

    return vectorizer_meta1, vectorizer_vector, model


def preprocess_data_for_predict(df, vectorizer_meta, vectorizer_vector):
    try:
        x1 = vectorizer_meta.transform(df['meta1'])
        x2 = vectorizer_vector.transform(df['vector'])
    except ValueError:
        raise ValueError
    vectorized_data = hstack((x1, x2))

    return vectorized_data


def predict(df, vectorzr_meta, vectorzr_vector, model):
    vectorized_data = preprocess_data_for_predict(df, vectorzr_meta,
                                                  vectorzr_vector)
    pred = model.predict(vectorized_data)
    df = pd.concat([df['id'].reset_index(drop=True), pd.Series(pred)],
                   axis=1, ignore_index=False)
    df.rename(columns={0: "class"}, inplace=True)

    return df
