import json
import matplotlib.pyplot as plt
from lexical_diversity import lex_div

from common_resources_for_evaluators import jsonFilePaths, clean_text

agentCountsAndMtld = dict()

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Count number of agents
    agents = set()
    for entry in jsonData:
        agents.add(entry["role"])

    # Calculate MTLD score
    all_text = " ".join(entry["content"] for entry in jsonData)
    cleaned_text = clean_text(all_text)
    
    # Calculate MTLD (Measure of Textual Lexical Diversity)
    mtld_score = lex_div.mtld(cleaned_text.split())

    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    agentCountsAndMtld[filename] = [(len(agents) - 1), mtld_score]

print("Number of agents and MTLD scores are calculated for each JSON file.")

# Plot a scatter plot of agent count vs MTLD score
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
    for key, value in agentCountsAndMtld.items():
        if selection_type in key:
            label = key.split("-")[1]
            color = scatterColors[label]
            ax.scatter(value[0], value[1], color=color, s=100)

    ax.set_title(selection_titles[idx])
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("MTLD Score")
    ax.grid(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
fig.legend(handles=legend_elements, loc="upper right", title="LLM-based MAS approach")
fig.suptitle("Agent Count vs MTLD Score")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mtldScoreVsAgentCount.png")
print("Scatter plot of agent count vs MTLD score is saved as mtldScoreVsAgentCount.png")

# Measure the correlation between agent count and binary weighting score
import numpy as np
agent_counts = np.array([value[0] for value in agentCountsAndMtld.values()])
mtld_scores = np.array([value[1] for value in agentCountsAndMtld.values()])

correlation = np.corrcoef(agent_counts, mtld_scores)[0, 1]
print(f"Correlation between agent count and MTLD score: {correlation:.2f}")
if abs(correlation) > 0.5:
    conclusion = "There is a strong correlation between agent count and MTLD score."
else:
    conclusion = "There is a weak correlation between agent count and MTLD score."
# Save the conclusion to a text file
with open("mtld_vs_agent_count_conclusion.txt", "w") as conclusion_file:
    conclusion_file.write(f"Correlation between agent count and MTLD score: {correlation:.2f}\n")
    conclusion_file.write(conclusion)