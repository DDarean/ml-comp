import pickle

import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn


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
            pickle.dump(self, f)

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

    def transform_data(self, df, vect_meta='', vect_vector='', load=False):
        try:
            meta1 = df['meta1']
            vector = df['vector']
        except ValueError:
            raise ValueError
        if load:
            with open(vect_meta, 'rb') as f:
                self.vectorizer_meta = pickle.load(f)
            with open(vect_vector, 'rb') as f:
                self.vectorizer_vector = pickle.load(f)
        x_meta = self.vectorizer_meta.transform(meta1)
        x_vector = self.vectorizer_vector.transform(vector)
        data = hstack((x_meta, x_vector))
        return data


class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs['input_shape'], out_features=5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=1000),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1000, out_features=5000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=5000, out_features=kwargs['input_shape'])
        )

    def encode(self, sample):
        return self.encoder(sample)

    def forward(self, sample):
        encoded = self.encoder(sample)
        decoded = self.decoder(encoded)
        return decoded


class TrainerAE:
    def __init__(self, df):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_shape=df.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.data = df.toarray()

    def fit_save_model(self, epochs=2, modelname='AEmodel'):
        train_loader = torch.utils.data.DataLoader(
            torch.from_numpy(self.data), batch_size=128, shuffle=True,
            num_workers=8, pin_memory=True
        )
        for epoch in range(epochs):
            self.model.train()
            for i, X_batch in enumerate(train_loader):
                X_batch = X_batch.float()
                self.optimizer.zero_grad()
                decoded = self.model(X_batch.to(self.device))
                loss = self.criterion(decoded, X_batch.to(self.device))
                loss.backward()
                self.optimizer.step()
        torch.save(self.model.state_dict(), f'models/{modelname}')

    def encode(self, data):
        data = data.toarray()
        self.model.eval()
        encoded = self.model.encode(torch.from_numpy(data).float().to(self.device))
        detached = encoded.cpu().detach().numpy()
        return detached
