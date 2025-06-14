import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

def extract_features(input_csv, output_csv):
    """
    Extracts behavioral features for each unique address from the given transaction CSV file.
    Saves the resulting features as a new CSV file.
    """
    # Load data
    df = pd.read_csv(input_csv, parse_dates=['Timestamp'])

    # Collect all unique addresses from both sender and receiver columns
    senders = set(df['Sender_Address'].unique())
    receivers = set(df['Receiver_Address'].unique())
    all_addresses = senders.union(receivers)

    features = []

    # For each address, calculate features
    for address in all_addresses:
        # Transactions where the address is a sender
        sent = df[df['Sender_Address'] == address]
        # Transactions where the address is a receiver
        received = df[df['Receiver_Address'] == address]
        # All transactions involving the address
        all_tx = pd.concat([sent, received]).sort_values('Timestamp')

        # Basic counts
        tx_count = len(all_tx)
        tx_sent_count = len(sent)
        tx_received_count = len(received)

        # Amounts
        total_sent = sent['Amount'].sum()
        total_received = received['Amount'].sum()
        net_balance_change = total_received - total_sent

        # Unique days of activity
        activity_days = all_tx['Timestamp'].dt.date.nunique()

        # Time-based features
        if not all_tx.empty:
            first_tx_time = all_tx['Timestamp'].min()
            last_tx_time = all_tx['Timestamp'].max()
            now = datetime.now(timezone.utc)
            # Ensure both datetimes are timezone-aware
            if first_tx_time.tzinfo is None:
                first_tx_time = first_tx_time.tz_localize('UTC')
            if last_tx_time.tzinfo is None:
                last_tx_time = last_tx_time.tz_localize('UTC')
            time_since_first_tx = (now - first_tx_time).days
            time_since_last_tx = (now - last_tx_time).days

            # Transaction intervals (in seconds)
            tx_times = all_tx['Timestamp'].sort_values().values
            if len(tx_times) > 1:
                intervals = np.diff(tx_times).astype('timedelta64[s]').astype(int)
                tx_interval_mean = np.mean(intervals)
                burstiness_score = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            else:
                tx_interval_mean = 0
                burstiness_score = 0

            # Entropy of activity by day of week
            days_of_week = all_tx['Timestamp'].dt.dayofweek.value_counts(normalize=True)
            entropy_active_time_windows = -np.sum(days_of_week * np.log2(days_of_week))
        else:
            first_tx_time = last_tx_time = pd.NaT
            time_since_first_tx = time_since_last_tx = 0
            tx_interval_mean = burstiness_score = entropy_active_time_windows = 0

        # Average transaction fee (if available)
        avg_fee = all_tx['Transaction_Fee'].mean() if 'Transaction_Fee' in all_tx else 0

        # Number of unique counterparties
        unique_counterparties = pd.concat([
            sent['Receiver_Address'],
            received['Sender_Address']
        ]).nunique()

        # Largest single transaction
        largest_tx = all_tx['Amount'].max()

        features.append({
            'Address': address,
            'tx_count': tx_count,
            'tx_sent_count': tx_sent_count,
            'tx_received_count': tx_received_count,
            'total_sent': total_sent,
            'total_received': total_received,
            'net_balance_change': net_balance_change,
            'activity_days': activity_days,
            'first_tx_time': first_tx_time,
            'last_tx_time': last_tx_time,
            'time_since_first_tx': time_since_first_tx,
            'time_since_last_tx': time_since_last_tx,
            'tx_interval_mean': tx_interval_mean,
            'burstiness_score': burstiness_score,
            'entropy_active_time_windows': entropy_active_time_windows,
            'avg_fee': avg_fee,
            'unique_counterparties': unique_counterparties,
            'largest_tx': largest_tx
        })

    # Create DataFrame from features
    features_df = pd.DataFrame(features)

    # Save features to CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Features extracted and saved to {output_csv}")

if __name__ == "__main__":
    # Define input and output paths for BTC and ETH
    btc_input = "../../data/btc/btc_data.csv"
    btc_output = "../../data/btc/btc_features.csv"
    eth_input = "../../data/eth/eth_data.csv"
    eth_output = "../../data/eth/eth_features.csv"

    # Extract features for BTC
    print("Extracting BTC features...")
    extract_features(btc_input, btc_output)

    # Extract features for ETH
    print("Extracting ETH features...")
    extract_features(eth_input, eth_output)