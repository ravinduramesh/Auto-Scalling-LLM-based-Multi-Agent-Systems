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
    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    topical_consistency_scores[filename] = topical_consistency

# Sort by topical consistency score
sorted_tc_items = sorted(topical_consistency_scores.items(), key=lambda x: x[1], reverse=True)
sorted_tc_labels = [item[0] for item in sorted_tc_items]
sorted_tc_scores = [item[1] for item in sorted_tc_items]
sorted_tc_colors = []
for label in sorted_tc_labels:
    if label.split("-")[1].startswith("autogen"):
        sorted_tc_colors.append('orangered')
    elif label.split("-")[1].startswith("DRTAG"):
        sorted_tc_colors.append('lawngreen')
    else:
        sorted_tc_colors.append('dodgerblue')

plt.figure(figsize=(25, 12))
plt.rcParams.update({'font.size': 15})
bars = plt.bar(sorted_tc_labels, sorted_tc_scores, color=sorted_tc_colors)
plt.xticks(rotation=90)
plt.ylabel("Topical Consistency (Avg. Pairwise Cosine Similarity)")
plt.title("Conversation Topical Consistency (BERTScore)")

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


# Statistical analysis with Mann-Whitney U rank test on MTLD scores
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta

standardSignificanceLevel = 0.05
conclusions = []

# Group BERT scores by label
autogen_scores = [score for label, score in topical_consistency_scores.items() if label.split("-")[1].startswith("autogen")]
drtag_scores = [score for label, score in topical_consistency_scores.items() if label.split("-")[1].startswith("DRTAG")]
iaag_scores = [score for label, score in topical_consistency_scores.items() if label.split("-")[1].startswith("IAAG")]

# Mann-Whitney U rank test to check if DRTAG is better than Autogen
stat, p = mannwhitneyu(drtag_scores, autogen_scores, alternative='less')
delta_value, delta_magnitude = cliffs_delta(drtag_scores, autogen_scores)
conclusions.append("DRTAG's topical consistency scores are lower than Autogen's topical consistency scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG has lower topical consistency than discussions generated using Autogen. This means that DRTAG discussions explore the topic with more depth and breadth in each dialogue, leading to a more diverse set of utterances.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG has lower topical consistency than discussions generated using Autogen. This means that DRTAG discussions do not necessarily explore the topic with more depth and breadth in each dialogue compared to Autogen discussions.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG is better than Autogen
stat, p = mannwhitneyu(iaag_scores, autogen_scores, alternative='less')
delta_value, delta_magnitude = cliffs_delta(iaag_scores, autogen_scores)
conclusions.append("IAAG's topical consistency scores are lower than Autogen's topical consistency scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG has lower topical consistency than discussions generated using Autogen. This means that IAAG discussions explore the topic with more depth and breadth in each dialogue, leading to a more diverse set of utterances.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG has lower topical consistency than discussions generated using Autogen. This means that IAAG discussions do not necessarily explore the topic with more depth and breadth in each dialogue compared to Autogen discussions.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG is better than IAAG
stat, p = mannwhitneyu(drtag_scores, iaag_scores, alternative='less')
delta_value, delta_magnitude = cliffs_delta(drtag_scores, iaag_scores)
conclusions.append("DRTAG's topical consistency scores are lower than IAAG's topical consistency scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG has lower topical consistency than discussions generated using IAAG. This means that DRTAG discussions explore the topic with more depth and breadth in each dialogue, leading to a more diverse set of utterances.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG has lower topical consistency than discussions generated using IAAG. This means that DRTAG discussions do not necessarily explore the topic with more depth and breadth in each dialogue compared to IAAG discussions.")
conclusions.append("")

# Save conclusions to a text file
with open("selfTopicalConsistencyConclusions.txt", "w") as conclusionsFile:
    for conclusion in conclusions:
        conclusionsFile.write(conclusion + "\n")
print("Topical consistency conclusions are saved as selfTopicalConsistencyConclusions.txt.")
