import pandas as pd

def generate_network_aggregates(input_path, output_path,
                                timestamp_col='Timestamp',
                                sender_col='Sender_Address',
                                receiver_col='Receiver_Address',
                                fee_col='Transaction_Fee'):
    # Load transaction data
    df = pd.read_csv(input_path)

    # Convert timestamp to date
    df['Date'] = pd.to_datetime(df[timestamp_col]).dt.date

    # Count daily transactions
    tx_count = df.groupby('Date').size().rename('tx_count')

    # Sum of transaction fees per day
    total_fees = df.groupby('Date')[fee_col].sum().rename('total_fees')

    # Count unique active addresses per day (senders and receivers)
    senders = df[['Date', sender_col]].rename(columns={sender_col: 'address'})
    receivers = df[['Date', receiver_col]].rename(columns={receiver_col: 'address'})
    all_addresses = pd.concat([senders, receivers], ignore_index=True)
    active_addresses = all_addresses.groupby('Date')['address'].nunique().rename('active_addresses')

    # Combine metrics
    daily_stats = pd.concat([tx_count, total_fees, active_addresses], axis=1).reset_index()

    # Save
    daily_stats.to_csv(output_path, index=False)
    print(f"Saved daily network statistics to: {output_path}")


if __name__ == "__main__":
    generate_network_aggregates(
        input_path="../../data/btc/btc_data.csv",
        output_path="../../data/btc/btc_network_daily.csv"
    )

    generate_network_aggregates(
        input_path="../../data/eth/eth_data.csv",
        output_path="../../data/eth/eth_network_daily.csv"
    )