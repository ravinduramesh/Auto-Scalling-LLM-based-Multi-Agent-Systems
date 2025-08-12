import json
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

thematicRelevanceAndTfIdf = dict()

# Initialize BERT model and vocabulary embedding
print("Loading BERT model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
vocab_string = ". ".join(ground_truth_vocab)
vocab_embedding = bert_model.encode([vocab_string], convert_to_numpy=True, show_progress_bar=False)[0]

# Get tfidf values for each document
tfidfTable = pd.read_csv('tfidf-table.csv')
tfidfSums = dict()
for column in tfidfTable.columns[1:]:
    tfidfSums[column] = tfidfTable[column].sum()

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Calculate thematic relevance using BERT embeddings
    all_text = " ".join(entry["content"] for entry in jsonData)
    cleaned_text = clean_text(all_text)
    text_embedding = bert_model.encode([cleaned_text], convert_to_numpy=True, show_progress_bar=False)
    similarity = cosine_similarity(text_embedding, vocab_embedding.reshape(1, -1)).flatten()
    thematic_relevance = float(np.mean(similarity))
    
    # Get TF-IDF sum value
    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    tfidf_sum_value = tfidfSums[filename]

    thematicRelevanceAndTfIdf[filename] = [thematic_relevance, tfidf_sum_value]

print("Thematic relevance scores and TF-IDF sums are calculated for each JSON file.")

# Plot a scatter plot of thematic relevance vs TF-IDF sum
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
plt.rcParams.update({'font.size': 15})

scatterColors = {
    "autogen": 'orangered',
    "DRTAG": 'lawngreen',
    "IAAG": 'dodgerblue'
}

# Define the selection types
selection_types = ["llm-selection", "round-robin-selection", "random-selection"]
selection_titles = ["LLM Selection", "Round Robin Selection", "Random Selection"]

for idx, selection_type in enumerate(selection_types):
    ax = axes[idx]
    for key, value in thematicRelevanceAndTfIdf.items():
        if selection_type in key:
            label = key.split("-")[1]
            color = scatterColors[label]
            ax.scatter(value[0], value[1], color=color, s=100)

    ax.set_title(selection_titles[idx])
    ax.set_xlabel("Thematic Relevance Score")
    ax.set_ylabel("TF-IDF Sum")
    ax.grid(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
fig.legend(handles=legend_elements, loc="upper right", title="LLM-based MAS approach")
fig.suptitle("Thematic Relevance vs TF-IDF Sum")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("thematicRelevanceVsTfIdf.png")
print("Scatter plot of thematic relevance vs TF-IDF sum is saved as thematicRelevanceVsTfIdf.png")

# Measure the correlation between thematic relevance and TF-IDF sum
thematic_relevance_scores = np.array([value[0] for value in thematicRelevanceAndTfIdf.values()])
tfidf_sums = np.array([value[1] for value in thematicRelevanceAndTfIdf.values()])

correlation = np.corrcoef(thematic_relevance_scores, tfidf_sums)[0, 1]
print(f"Correlation between thematic relevance and TF-IDF sum: {correlation:.2f}")
if abs(correlation) > 0.5:
    conclusion = "There is a strong correlation between thematic relevance and TF-IDF sum."
else:
    conclusion = "There is a weak correlation between thematic relevance and TF-IDF sum."

# Save the conclusion to a text file
with open("thematic_relevance_vs_tfidf_conclusion.txt", "w") as conclusion_file:
    conclusion_file.write(f"Correlation between thematic relevance and TF-IDF sum: {correlation:.2f}\n")
    conclusion_file.write(conclusion)

print("Analysis complete. Results saved to file.")
