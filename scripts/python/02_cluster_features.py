import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def cluster_and_score(input_csv, output_csv, n_clusters=6, use_dbscan=False):
    """
    Loads address features, performs scaling, clustering, anomaly scoring, and dimensionality reduction.
    Saves the results to a new CSV file.
    """
    # Load features
    df = pd.read_csv(input_csv)
    address_col = 'Address'
    feature_cols = [col for col in df.columns if col not in [address_col, 'first_tx_time', 'last_tx_time']]
    X = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    if use_dbscan:
        clusterer = DBSCAN(eps=1.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(X_scaled)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X_scaled)
    df['cluster_id'] = cluster_labels

    # Anomaly scoring (Isolation Forest)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly_score'] = -iso_forest.fit_predict(X_scaled)  # 1=normal, -1=anomaly

    # Local Outlier Factor (optional, can be commented out)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_scores = lof.fit_predict(X_scaled)
    df['lof_anomaly'] = -lof_scores  # 1=normal, -1=anomaly

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Clustering and anomaly scores saved to {output_csv}")

if __name__ == "__main__":
    # BTC
    btc_input = "../../data/btc/btc_features.csv"
    btc_output = "../../data/btc/btc_clusters.csv"
    print("Processing BTC...")
    cluster_and_score(btc_input, btc_output, n_clusters=6, use_dbscan=False)

    # ETH
    eth_input = "../../data/eth/eth_features.csv"
    eth_output = "../../data/eth/eth_clusters.csv"
    print("Processing ETH...")
    cluster_and_score(eth_input, eth_output, n_clusters=6, use_dbscan=False)