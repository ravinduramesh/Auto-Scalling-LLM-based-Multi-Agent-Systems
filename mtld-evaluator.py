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
    filename = jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]
    mtld_scores[filename] = mtld_score

# Sort by MTLD score
sorted_items = sorted(mtld_scores.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [item[0] for item in sorted_items]
sorted_scores = [item[1] for item in sorted_items]
sorted_colors = []
for label in sorted_labels:
    if label.split("\n")[1].startswith("autogen"):
        sorted_colors.append('orangered')
    elif label.split("\n")[1].startswith("DRTAG"):
        sorted_colors.append('lawngreen')
    else:
        sorted_colors.append('dodgerblue')

plt.figure(figsize=(25, 10))
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
import numpy as np
from scipy.stats import mannwhitneyu
import scikit_posthocs as sp

standardSignificanceLevel = 0.05
conclusions = []

# Group MTLD scores by label
autogen_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen")]
drtag_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG")]
iaag_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG")]

autogen_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-llm-selection")]
drtag_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-llm-selection")]
iaag_llm_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-llm-selection")]
autogen_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-random-selection")]
drtag_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-random-selection")]
iaag_random_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-random-selection")]
autogen_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("autogen-round-robin-selection")]
drtag_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("DRTAG-round-robin-selection")]
iaag_round_robin_selection_scores = [score for label, score in mtld_scores.items() if label.split("\n")[1].startswith("IAAG-round-robin-selection")]

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

# Mann-Whitney U rank test to check if DRTAG LLM selection is better than Autogen LLM selection
stat, p = mannwhitneyu(drtag_llm_selection_scores, autogen_llm_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG LLM selection's MTLD scores are better than Autogen LLM selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG LLM selection discuss the topic in broader and deeper way than discussions generated using Autogen LLM selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG LLM selection discuss the topic in broader and deeper way than discussions generated using Autogen LLM selection.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG random selection is better than Autogen random selection
stat, p = mannwhitneyu(drtag_random_selection_scores, autogen_random_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG random selection's MTLD scores are better than Autogen random selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG random selection discuss the topic in broader and deeper way than discussions generated using Autogen random selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG random selection discuss the topic in broader and deeper way than discussions generated using Autogen random selection.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG round robin selection is better than Autogen round robin selection
stat, p = mannwhitneyu(drtag_round_robin_selection_scores, autogen_round_robin_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG round robin selection's MTLD scores are better than Autogen round robin selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG round robin selection discuss the topic in broader and deeper way than discussions generated using Autogen round robin selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG round robin selection discuss the topic in broader and deeper way than discussions generated using Autogen round robin selection.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG LLM selection is better than Autogen LLM selection
stat, p = mannwhitneyu(iaag_llm_selection_scores, autogen_llm_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG LLM selection's MTLD scores are better than Autogen LLM selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG LLM selection discuss the topic in broader and deeper way than discussions generated using Autogen LLM selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG LLM selection discuss the topic in broader and deeper way than discussions generated using Autogen LLM selection.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG random selection is better than Autogen random selection
stat, p = mannwhitneyu(iaag_random_selection_scores, autogen_random_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG random selection's MTLD scores are better than Autogen random selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG random selection discuss the topic in broader and deeper way than discussions generated using Autogen random selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG random selection discuss the topic in broader and deeper way than discussions generated using Autogen random selection.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG round robin selection is better than Autogen round robin selection
stat, p = mannwhitneyu(iaag_round_robin_selection_scores, autogen_round_robin_selection_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG round robin selection's MTLD scores are better than Autogen round robin selection's MTLD scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG round robin selection discuss the topic in broader and deeper way than discussions generated using Autogen round robin selection.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG round robin selection discuss the topic in broader and deeper way than discussions generated using Autogen round robin selection.")
conclusions.append("")

# Save conclusions to a text file
with open("mtld-results-analysis-conclusions.txt", "w") as file:
    for conclusion in conclusions:
        file.write(conclusion + "\n")
print("All conclusions are written to the file 'mtld-results-analysis-conclusions.txt'.")
