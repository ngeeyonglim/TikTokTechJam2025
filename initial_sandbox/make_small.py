import pandas as pd

# Path to your MNIST train CSV
input_file = "mnist_train.csv"
output_file = "mnist_train_small.csv"

# Load CSV
df = pd.read_csv(input_file)

# Take a random 1/3 of the rows
df_random_third = df.sample(frac=1/3, random_state=42)  # random_state makes it reproducible

# Save to new CSV
df_random_third.to_csv(output_file, index=False)

print(f"Saved random 1/3 rows ({len(df_random_third)} out of {len(df)}) to {output_file}")
