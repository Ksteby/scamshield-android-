import pandas as pd


dataset_path = "data/italian_sms.csv"

# Caricamento del file CSV
df = pd.read_csv(dataset_path)

print("Numero di messaggi :", len(df))

print("\nColumns :")
print(df.columns)

print("\nPremi righe :")
print(df.head())

print("\nRipartizione delle classe :")
print(df["labels"].value_counts())