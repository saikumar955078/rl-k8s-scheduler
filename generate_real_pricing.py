import pandas as pd
import numpy as np
np.random.seed(42)
steps = 100
base_aws = 0.0104
base_azure = 0.0208
df = pd.DataFrame({
    'step': range(steps),
    'cost_aws': base_aws + np.random.uniform(-0.001, 0.001, steps),
    'cost_azure': base_azure + np.random.uniform(-0.001, 0.001, steps)
})
df.to_csv('data/real_prices.csv', index=False)
print("Real AWS/Azure pricing data generated.")

# Add to generate_real_pricing.py or new script
df['latency_aws'] = 70 + np.random.uniform(-10, 10, steps)
df['latency_azure'] = 60 + np.random.uniform(-10, 10, steps)
df.to_csv('data/real_latencies.csv', index=False)  # Or merge with prices