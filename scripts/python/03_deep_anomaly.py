import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(X, n_epochs=50, batch_size=128, lr=1e-3, latent_dim=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(X.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(X):.6f}")
    return model

def compute_reconstruction_loss(model, X):
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        loss = ((recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()
    return loss

def process_chain(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    address_col = 'Address'
    # Select only numerical features (exclude labels, dates, PCA, cluster_id, anomaly_score)
    exclude = [address_col, 'first_tx_time', 'last_tx_time', 'cluster_id', 'anomaly_score', 'lof_anomaly', 'pca_1', 'pca_2']
    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].fillna(0).values

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train autoencoder only on "normal" addresses (anomaly_score == 1)
    if 'anomaly_score' in df.columns:
        normal_mask = df['anomaly_score'] == 1
    else:
        normal_mask = np.ones(len(df), dtype=bool)
    X_train = X_scaled[normal_mask]

    print(f"Training autoencoder on {X_train.shape[0]} normal samples...")
    model = train_autoencoder(X_train, n_epochs=50, batch_size=128, lr=1e-3, latent_dim=8)

    # Compute reconstruction loss for all addresses
    print("Computing reconstruction loss for all addresses...")
    recon_loss = compute_reconstruction_loss(model, X_scaled)
    df['deep_anomaly_score'] = recon_loss

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Saved deep anomaly scores to {output_csv}")

if __name__ == "__main__":
    # BTC
    btc_input = "../../data/btc/btc_clusters.csv"
    btc_output = "../../data/btc/btc_deep_anomaly.csv"
    print("Processing BTC...")
    process_chain(btc_input, btc_output)

    # ETH
    eth_input = "../../data/eth/eth_clusters.csv"
    eth_output = "../../data/eth/eth_deep_anomaly.csv"
    print("Processing ETH...")
    process_chain(eth_input, eth_output)