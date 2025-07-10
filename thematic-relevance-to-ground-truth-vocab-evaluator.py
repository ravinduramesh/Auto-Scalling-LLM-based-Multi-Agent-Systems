import json
import matplotlib.pyplot as plt
from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_text_from_json(jsonFilePath):
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)
    all_text = " ".join(entry["content"] for entry in jsonData)
    return all_text

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
vocab_string = ". ".join(ground_truth_vocab)

thematic_relevance_scores = dict()

for jsonFilePath in jsonFilePaths:
    text = get_text_from_json(jsonFilePath)
    cleaned_text = clean_text(text)
    
    # Encode all utterances and the vocab string
    utterance_embeddings = bert_model.encode([cleaned_text], convert_to_numpy=True, show_progress_bar=False)
    vocab_embedding = bert_model.encode([vocab_string], convert_to_numpy=True, show_progress_bar=False)[0]
    # Calculate cosine similarity between each utterance and the vocab embedding
    similarities = cosine_similarity(utterance_embeddings, vocab_embedding.reshape(1, -1)).flatten()
    thematic_relevance = float(np.mean(similarities))

    filename = 'conv' + jsonFilePath.split('/')[-2][-2:] + '-' + jsonFilePath.split('/')[-1]
    thematic_relevance_scores[filename] = thematic_relevance

# Sort and plot
sorted_tc_items = sorted(thematic_relevance_scores.items(), key=lambda x: x[1], reverse=True)
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

plt.figure(figsize=(35, 12))
plt.rcParams.update({'font.size': 15})
bars = plt.bar(sorted_tc_labels, sorted_tc_scores, color=sorted_tc_colors)
plt.xticks(rotation=90)
plt.ylabel("Thematic Relevance")
plt.title("Conversation Thematic Relevance to The Ground Truth Vocabulary (BERT Embeddings)")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
plt.legend(handles=legend_elements, loc='upper right', title="LLM-based MAS approach")

plt.tight_layout()
plt.savefig("thematicRelevanceToGroundTruthVocab.png")
print("Thematic relevance to ground truth vocabulary graph is plotted successfully.")


# Statistical analysis with Mann-Whitney U rank test on thematic relevance scores
import numpy as np
from scipy.stats import mannwhitneyu
import scikit_posthocs as sp

standardSignificanceLevel = 0.05
conclusions = []

# Group thematic relevance scores by label
autogen_scores = [score for label, score in thematic_relevance_scores.items() if "autogen" in label]
drtag_scores = [score for label, score in thematic_relevance_scores.items() if "DRTAG" in label]
iaag_scores = [score for label, score in thematic_relevance_scores.items() if "IAAG" in label]

# autogen_llm_selection_scores = [score for label, score in thematic_relevance_scores.items() if "autogen-llm-selection" in label]
# drtag_llm_selection_scores = [score for label, score in thematic_relevance_scores.items() if "DRTAG-llm-selection" in label]
# iaag_llm_selection_scores = [score for label, score in thematic_relevance_scores.items() if "IAAG-llm-selection" in label]
# autogen_random_selection_scores = [score for label, score in thematic_relevance_scores.items() if "autogen-random-selection" in label]
# drtag_random_selection_scores = [score for label, score in thematic_relevance_scores.items() if "DRTAG-random-selection" in label]
# iaag_random_selection_scores = [score for label, score in thematic_relevance_scores.items() if "IAAG-random-selection" in label]
# autogen_round_robin_selection_scores = [score for label, score in thematic_relevance_scores.items() if "autogen-round-robin" in label]
# drtag_round_robin_selection_scores = [score for label, score in thematic_relevance_scores.items() if "DRTAG-round-robin" in label]
# iaag_round_robin_selection_scores = [score for label, score in thematic_relevance_scores.items() if "IAAG-round-robin" in label]

# Mann-Whitney U rank test to check if DRTAG is better than Autogen
stat, p = mannwhitneyu(drtag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's Thematic Relevance scores are better than Autogen's Thematic Relevance scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG are more thematically relavant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG are more thematically relavant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG is better than Autogen
stat, p = mannwhitneyu(iaag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG's Thematic Relevance scores are better than Autogen's Thematic Relevance scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG are more thematically relavant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG are more thematically relavant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG is better than IAAG
stat, p = mannwhitneyu(drtag_scores, iaag_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's Thematic Relevance scores are better than IAAG's Thematic Relevance scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG are more thematically relavant to the scenario than discussions generated using IAAG.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG are more thematically relavant to the scenario than discussions generated using IAAG.")
conclusions.append("")

# Save conclusions to a text file
with open("thematic_relevance_conclusions.txt", "w") as f:
    for conclusion in conclusions:
        f.write(conclusion + "\n")
print("Thematic relevance conclusions saved to thematic_relevance_conclusions.txt.")
