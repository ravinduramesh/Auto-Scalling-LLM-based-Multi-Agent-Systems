import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from lexical_diversity import lex_div as ld

from common_resources_for_evaluators import jsonFilePaths, clean_text

def get_text_from_json(jsonFilePath):
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)
    all_text = " ".join(entry["content"] for entry in jsonData)
    return all_text

mtld_scores = dict()

for jsonFilePath in jsonFilePaths:
    text = get_text_from_json(jsonFilePath)
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    mtld_score = ld.mtld(tokens)
    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    mtld_scores[filename] = mtld_score

# Sort by MTLD score
sorted_items = sorted(mtld_scores.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [item[0] for item in sorted_items]
sorted_scores = [item[1] for item in sorted_items]
sorted_colors = []
for label in sorted_labels:
    if label.split("-")[1].startswith("autogen"):
        sorted_colors.append('orangered')
    elif label.split("-")[1].startswith("DRTAG"):
        sorted_colors.append('lawngreen')
    else:
        sorted_colors.append('dodgerblue')

plt.figure(figsize=(35, 12))
plt.rcParams.update({'font.size': 15})

bars = plt.bar(sorted_labels, sorted_scores, color=sorted_colors)
plt.xticks(rotation=90)
plt.ylabel("MTLD Score")
plt.title("Conversation MTLD (Measure of Textual Lexical Diversity) Scores")

# Create custom legend
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
plt.legend(handles=legend_elements, loc='upper right', title="LLM-based MAS approach")

plt.tight_layout()
plt.savefig("mtldScores.png")
print("MTLD graph is plotted successfully.")


# Statistical analysis with Mann-Whitney U rank test on MTLD scores
from scipy.stats import mannwhitneyu

standardSignificanceLevel = 0.05
conclusions = []

# Group MTLD scores by label
autogen_scores = [score for label, score in mtld_scores.items() if label.split("-")[1].startswith("autogen")]
drtag_scores = [score for label, score in mtld_scores.items() if label.split("-")[1].startswith("DRTAG")]
iaag_scores = [score for label, score in mtld_scores.items() if label.split("-")[1].startswith("IAAG")]

# autogen_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-llm-selection")]
# drtag_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-llm-selection")]
# iaag_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-llm-selection")]
# autogen_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-random-selection")]
# drtag_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-random-selection")]
# iaag_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-random-selection")]
# autogen_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-round-robin-selection")]
# drtag_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-round-robin-selection")]
# iaag_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-round-robin-selection")]

# Mann-Whitney U rank test to check if DRTAG is better than Autogen
stat, p = mannwhitneyu(drtag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's MTLD scores are better than Autogen's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG discuss the topic in broader and deeper way than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG discuss the topic in broader and deeper way than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG is better than Autogen
stat, p = mannwhitneyu(iaag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG's MTLD scores are better than Autogen's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG discuss the topic in broader and deeper way than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG discuss the topic in broader and deeper way than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG is better than IAAG
stat, p = mannwhitneyu(drtag_scores, iaag_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's MTLD scores are better than IAAG's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG discuss the topic in broader and deeper way than discussions generated using IAAG.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG discuss the topic in broader and deeper way than discussions generated using IAAG.")
conclusions.append("")

#Save conclusions to a text file
with open("mtld_conclusions.txt", "w") as file:
    for conclusion in conclusions:
        file.write(conclusion + "\n")
print("MTLD conclusions saved to mtld_conclusions.txt.")
