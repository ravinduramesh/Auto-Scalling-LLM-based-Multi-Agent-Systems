#import tfidf-table.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("tfidf-table.csv")
sums = data.iloc[:, 1:].sum(axis=0)

plt.figure(figsize=(10, 10))
plt.bar(data.columns[1:], sums)
plt.xticks(rotation=90)
plt.ylabel("TF-IDF Sum")
plt.title("TF-IDF Sum for each conversation")
plt.tight_layout()
plt.savefig("tf-idf-sums.png")