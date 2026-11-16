
import pandas as pd
import os

# Load dataset
df = pd.read_csv("dataset/malicious_phish.csv")

print("Original dataset:", df.shape)

# Remove missing values
df = df.dropna(subset=["url", "type"])

# Remove duplicate URLs
df = df.drop_duplicates(subset=["url"])

# Remove empty URLs
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""]

# Remove invalid category records
valid_types = ["benign", "phishing", "defacement", "malware"]
df = df[df["type"].isin(valid_types)]

# Save cleaned dataset
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/cleaned_urls.csv", index=False)

print("\nData cleaning completed!")
print("Cleaned dataset:", df.shape)

print("\nCategory distribution:")
print(df["type"].value_counts())

print("\nSaved to dataset/cleaned_urls.csv")
