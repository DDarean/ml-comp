from torch import nn
import torch


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

    def fit_save_model(self, epochs=1, modelname='AEmodel'):
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
        torch.save(self.model, f'models/{modelname}')

    def encode(self, data):
        data = data.toarray()
        self.model.eval()
        encoded = self.model.encode(torch.from_numpy(data).float().to(self.device))
        detached = encoded.cpu().detach().numpy()
        return detached
