# Based on paper https://arxiv.org/abs/1806.02920

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import tqdm

class GAINDataset(Dataset):

  def __init__(self, X, M):
    self.X = torch.from_numpy(X.astype('float32'))
    self.M = torch.from_numpy(M.astype('float32'))

  def __getitem__(self, index):
    return self.X[index], self.M[index]

  def __len__(self):
    return len(self.X)

class GAINGenerator(nn.Module):

  def __init__(self, n_features):
    super().__init__()
    self.layer1 = nn.Linear(2*n_features, n_features)
    self.layer2 = nn.Linear(n_features, n_features)
    self.layer3 = nn.Linear(n_features, n_features)

    nn.init.xavier_normal_(self.layer1.weight.data)
    nn.init.zeros_(self.layer1.bias.data)
    nn.init.xavier_normal_(self.layer2.weight.data)
    nn.init.zeros_(self.layer2.bias.data)
    nn.init.xavier_normal_(self.layer3.weight.data)
    nn.init.zeros_(self.layer3.bias.data)

  def forward(self, X, M):
    inputs = torch.concat([X,M], dim=1)
    a = nn.functional.relu(self.layer1(inputs))
    a = nn.functional.relu(self.layer2(a))
    a = nn.functional.relu(self.layer3(a))
    return a

class GAINDiscriminator(nn.Module):

  def __init__(self, n_features):
    super().__init__()
    self.layer1 = nn.Linear(2*n_features, n_features)
    self.layer2 = nn.Linear(n_features, n_features)
    self.layer3 = nn.Linear(n_features, n_features)

    nn.init.xavier_normal_(self.layer1.weight.data)
    nn.init.zeros_(self.layer1.bias.data)
    nn.init.xavier_normal_(self.layer2.weight.data)
    nn.init.zeros_(self.layer2.bias.data)
    nn.init.xavier_normal_(self.layer3.weight.data)
    nn.init.zeros_(self.layer3.bias.data)

  def forward(self, X, H):
    inputs = torch.concat([X,H], dim=1)
    a = nn.functional.relu(self.layer1(inputs))
    a = nn.functional.relu(self.layer2(a))
    a = nn.functional.relu(self.layer3(a))
    return a

class GAINModel:

  def __init__(self):
    self.scaler = MinMaxScaler()
    self.gen = None
    self.disc = None
    self.is_fitted = False

  def discriminator_loss(self, M_batch, X_new, H, alpha):
    G_sample = self.gen(X_new, M_batch)
    X_hat = X_new * M_batch + G_sample * (1 - M_batch)
    D_prob = self.disc(X_hat, H)
    D_loss = -torch.mean(M_batch * torch.log(D_prob + 1e-8) + (1 - M_batch) * torch.log(1. - D_prob + 1e-8))
    return D_loss

  def generator_loss(self, M_batch, X_new, H, alpha):
    G_sample = self.gen(X_new, M_batch)
    X_hat = X_new * M_batch + G_sample * (1 - M_batch)
    D_prob = self.disc(X_hat, H)
    G_loss = -torch.mean((1 - M_batch) * torch.log(D_prob + 1e-8)) + alpha * torch.mean((M_batch * X_new - M_batch * G_sample)**2) / torch.mean(M_batch)
    return G_loss

  def fit_impute(self, X_data, epochs=10, batch_size=32, hint_rate=0.8, alpha=500, gen_lr=1e-3, disc_lr=1e-3):

    n_features = X_data.shape[1]
    self.gen = GAINGenerator(n_features)
    self.disc = GAINDiscriminator(n_features)

    X = self.scaler.fit_transform(X_data)
    M = 1 - np.isnan(X)
    X = np.nan_to_num(X, 0)

    dataset = GAINDataset(X, M)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_G = torch.optim.SGD(self.gen.parameters(), lr=gen_lr)
    optimizer_D = torch.optim.SGD(self.disc.parameters(), lr=disc_lr)

    for i in range(epochs):

      print(f'------- Epoch {i+1} -------')

      for j, (X_batch, M_batch) in enumerate(tqdm.tqdm(dataloader)):

        Z = np.random.uniform(0, 0.01, size=X_batch.shape)
        Z = torch.tensor(Z, dtype=torch.float32)
        H = torch.tensor(1 * (np.random.uniform(0., 1., size=X_batch.shape) < hint_rate), dtype=torch.float32) * M_batch

        X_new = M_batch * X_batch + (1 - M_batch) * Z

        # Discriminator optimization
        optimizer_D.zero_grad()
        D_loss = self.discriminator_loss(M_batch, X_new, H, alpha)
        D_loss.backward()
        optimizer_D.step()

        # Generator optimization
        optimizer_G.zero_grad()
        G_loss = self.discriminator_loss(M_batch, X_new, H, alpha)
        G_loss.backward()
        optimizer_G.step()

    self.is_fitted = True

    X = torch.tensor(X, dtype=torch.float32)
    M = torch.tensor(M, dtype=torch.float32)
    Z_full = np.random.uniform(0, 0.01, size=X.shape)
    Z_full = torch.tensor(Z_full, dtype=torch.float32)

    imputed = self.gen(M * X + (1 - M) * Z_full, M)
    imputed = M * X + (1 - M) * imputed
    imputed = self.scaler.inverse_transform(imputed.detach().numpy())
    imputed = pd.DataFrame(imputed, columns=X_data.columns)
    return imputed

  def impute(self, X_new_data):

    if not self.is_fitted:
      raise Exception('GAN has not been fit to any data')

    X_new = self.scaler.transform(X_new_data)
    M_new = 1 - np.isnan(X_new)
    X_new = np.nan_to_num(X_new, 0)

    X_new = torch.tensor(X_new, dtype=torch.float32)
    M_new = torch.tensor(M_new, dtype=torch.float32)
    Z_new = np.random.uniform(0, 0.01, size=X_new.shape)
    Z_new = torch.tensor(Z_new, dtype=torch.float32)

    imputed = self.gen(M_new * X_new + (1 - M_new) * Z_new, M_new)
    imputed = M_new * X_new + (1 - M_new) * imputed
    imputed = self.scaler.inverse_transform(imputed.detach().numpy())
    imputed = pd.DataFrame(imputed, columns=X_new_data.columns)

    return imputed