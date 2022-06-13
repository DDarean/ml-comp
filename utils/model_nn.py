from torch import nn


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
