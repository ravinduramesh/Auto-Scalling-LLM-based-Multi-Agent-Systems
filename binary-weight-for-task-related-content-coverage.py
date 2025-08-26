import json
import matplotlib.pyplot as plt
from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

conversationScores = dict()
ground_truth_vocab = set(ground_truth_vocab)

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    initialAgents = {"Patient"}
    initialAgentsContent = ""
    newlyGeneratedAgentsContent = ""

    for entry in jsonData:
        if entry["role"] in initialAgents:
            initialAgentsContent += entry["content"] + " "
            initialAgents.remove(entry["role"])
        else:
            newlyGeneratedAgentsContent += entry["content"] + " "

    # Clean the text data and remove stopwords
    initialAgentsCleanedContent = clean_text(initialAgentsContent)
    newlyGeneratedAgentsCleanedContent = clean_text(newlyGeneratedAgentsContent)

    initialAgentsTermSet = set(initialAgentsCleanedContent.split())
    newlyGeneratedAgentsTermSet = set(newlyGeneratedAgentsCleanedContent.split())
    termsToScoreConversation = ground_truth_vocab.intersection(newlyGeneratedAgentsTermSet - initialAgentsTermSet)

    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    conversationScores[filename] = len(termsToScoreConversation)


sorted_items = sorted(conversationScores.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [item[0] for item in sorted_items]
sorted_scores = [item[1] for item in sorted_items]

# Assign colors based on label type for legend
barColors = []
color_map = {"autogen": "orangered", "DRTAG": "lawngreen", "IAAG": "dodgerblue"}
for label in sorted_labels:
    if label.split("-")[1].startswith("autogen"):
        barColors.append(color_map["autogen"])
    elif label.split("-")[1].startswith("DRTAG"):
        barColors.append(color_map["DRTAG"])
    else:
        barColors.append(color_map["IAAG"])

plt.figure(figsize=(25, 12))
plt.rcParams.update({'font.size': 15})

bars = plt.bar(sorted_labels, sorted_scores, color=barColors)
plt.xticks(rotation=90)
plt.ylabel("Binary Weighting Score")
plt.title("Task Related Content Coverage by All Agents")
plt.tight_layout()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map["autogen"], label='autogen'),
    Patch(facecolor=color_map["DRTAG"], label='DRTAG'),
    Patch(facecolor=color_map["IAAG"], label='IAAG')
]
plt.legend(handles=legend_elements, loc='upper right', title="Label Type")

plt.savefig("binaryWeightingScoresOfConversations2.png")
print("Graph is plotted successfully.")

# Statistical analysis with Mann-Whitney U rank test on binary weighting scores
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta

standardSignificanceLevel = 0.05
conclusions = []

# Group binary weighting scores by label
autogen_scores = [score for label, score in conversationScores.items() if "autogen" in label]
drtag_scores = [score for label, score in conversationScores.items() if "DRTAG" in label]
iaag_scores = [score for label, score in conversationScores.items() if "IAAG" in label]

# autogen_llm_selection_scores = [score for label, score in conversationScores.items() if "autogen-llm-selection" in label]
# drtag_llm_selection_scores = [score for label, score in conversationScores.items() if "DRTAG-llm-selection" in label]
# iaag_llm_selection_scores = [score for label, score in conversationScores.items() if "IAAG-llm-selection" in label]
# autogen_random_selection_scores = [score for label, score in conversationScores.items() if "autogen-random-selection" in label]
# drtag_random_selection_scores = [score for label, score in conversationScores.items() if "DRTAG-random-selection" in label]
# iaag_random_selection_scores = [score for label, score in conversationScores.items() if "IAAG-random-selection" in label]
# autogen_round_robin_selection_scores = [score for label, score in conversationScores.items() if "autogen-round-robin" in label]
# drtag_round_robin_selection_scores = [score for label, score in conversationScores.items() if "DRTAG-round-robin" in label]
# iaag_round_robin_selection_scores = [score for label, score in conversationScores.items() if "IAAG-round-robin" in label]

# Mann-Whitney U rank test to check if DRTAG is better than Autogen
stat, p = mannwhitneyu(drtag_scores, autogen_scores, alternative='greater')
delta_value, delta_magnitude = cliffs_delta(drtag_scores, autogen_scores)
conclusions.append("DRTAG's Binary Weighting scores are better than Autogen's Binary Weighting scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relavant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relavant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG is better than Autogen
stat, p = mannwhitneyu(iaag_scores, autogen_scores, alternative='greater')
delta_value, delta_magnitude = cliffs_delta(iaag_scores, autogen_scores)
conclusions.append("IAAG's Binary Weighting scores are better than Autogen's Binary Weighting scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG contains more keywords relavant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG contains more keywords relavant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG is better than IAAG
stat, p = mannwhitneyu(drtag_scores, iaag_scores, alternative='greater')
delta_value, delta_magnitude = cliffs_delta(drtag_scores, iaag_scores)
conclusions.append("DRTAG's Binary Weighting scores are better than IAAG's Binary Weighting scores")
conclusions.append(f"Mann-Whitney U Test: H={stat:.3f}, p={p:.4f}")
conclusions.append(f"Cliff's Delta: {delta_value:.3f} ({delta_magnitude})")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relavant to the scenario than discussions generated using IAAG.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relavant to the scenario than discussions generated using IAAG.")
conclusions.append("")

# Save conclusions to a text file
with open("binary_weighting_conclusions2.txt", "w") as file:
    for conclusion in conclusions:
        file.write(conclusion + "\n")
print("Binary weighting conclusions saved to binary_weighting_conclusions2.txt.")