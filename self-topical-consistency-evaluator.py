import json
import matplotlib.pyplot as plt
from common_resources_for_evaluators import jsonFilePaths, clean_text

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_contents_from_json(jsonFilePath):
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)
    return [clean_text(entry["content"]) for entry in jsonData]

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

topical_consistency_scores = dict()

for jsonFilePath in jsonFilePaths:
    contents = get_contents_from_json(jsonFilePath)
    if len(contents) < 2:
        topical_consistency = 1.0  # Only one text, so perfectly consistent
    else:
        embeddings = bert_model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        n = len(contents)
        triu_indices = np.triu_indices(n, k=1)
        avg_similarity = np.mean(sim_matrix[triu_indices])
        topical_consistency = avg_similarity
    filename = jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]
    topical_consistency_scores[filename] = topical_consistency

# Sort by topical consistency score
sorted_tc_items = sorted(topical_consistency_scores.items(), key=lambda x: x[1], reverse=True)
sorted_tc_labels = [item[0] for item in sorted_tc_items]
sorted_tc_scores = [item[1] for item in sorted_tc_items]
sorted_tc_colors = []
for label in sorted_tc_labels:
    if label.split("\n")[1].startswith("autogen"):
        sorted_tc_colors.append('orangered')
    elif label.split("\n")[1].startswith("DRTAG"):
        sorted_tc_colors.append('lawngreen')
    else:
        sorted_tc_colors.append('dodgerblue')

plt.figure(figsize=(35, 12))
plt.rcParams.update({'font.size': 15})
bars = plt.bar(sorted_tc_labels, sorted_tc_scores, color=sorted_tc_colors)
plt.xticks(rotation=90)
plt.ylabel("Topical Consistency (Avg. Pairwise Cosine Similarity)")
plt.title("Conversation Topical Consistency (BERT Embeddings)")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
plt.legend(handles=legend_elements, loc='upper right', title="LLM-based MAS approach")

plt.tight_layout()
plt.savefig("selfTopicalConsistencyScores.png")
print("Topical Consistency graph is plotted successfully.")