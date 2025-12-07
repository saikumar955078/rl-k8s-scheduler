import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSVs
prices = pd.read_csv('data/real_prices.csv')
latencies = pd.read_csv('data/real_latencies.csv')

# Extract average CPU from your Locust/Prometheus CSVs
aws_load = pd.read_csv('data/local_aws_load_stats.csv')
azure_load = pd.read_csv('data/local_azure_load_stats.csv')

# Assume Locust CSV has a column 'Average Response Time' as a proxy for load
# Or replace with Prometheus CPU column if you exported from Prometheus
aws_cpu = aws_load[['Average Response Time']].mean().rename({'Average Response Time':'cpu_aws'})
azure_cpu = azure_load[['Average Response Time']].mean().rename({'Average Response Time':'cpu_azure'})

# Combine into one DataFrame
df = pd.concat([
    prices,
    latencies[['latency_aws','latency_azure']],
    pd.DataFrame({'cpu_aws':[aws_cpu[0]], 'cpu_azure':[azure_cpu[0]]})
], axis=1)

# Normalize all columns between 0-1
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save processed CSV
normalized.to_csv('data/processed/normalized_rl_data.csv', index=False)

print("Normalized RL data saved to data/processed/normalized_rl_data.csv")
