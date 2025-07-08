import json
import pandas as pd
import matplotlib.pyplot as plt

from common_resources_for_evaluators import jsonFilePaths

agentCountsAndTfIdf = dict()

# Get tfidf values for each document
tfidfTable = pd.read_csv('tfidf-table.csv')
tfidfSums = dict()
for column in tfidfTable.columns[1:]:
    tfidfSums[column] = tfidfTable[column].sum()

for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath) as jsonFile:
        jsonData = json.load(jsonFile)
    
    # Count number of agents
    agents = set()
    for entry in jsonData:
        agents.add(entry["role"])
    
    # Get first tfidf value in tfidfSums
    tfidfSumValue = tfidfSums[jsonFilePath.split("/")[-2] + '/' + jsonFilePath.split("/")[-1]]

    agentCountsAndTfIdf[jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]] = [(len(agents) - 1), tfidfSumValue]

print("Number of agents are counted for each JSON file.")

# Plot a scatter plot of agent count vs binary weight
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
    for key, value in agentCountsAndTfIdf.items():
        if selection_type in key:
            label = (key.split("\n")[-1]).split("-")[0]
            color = scatterColors[label]
            ax.scatter(value[0], value[1], color=color, s=100)

    ax.set_title(selection_titles[idx])
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("TF-IDF Sum")
    ax.grid(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
fig.legend(handles=legend_elements, loc="upper right", title="LLM-based MAS approach")
fig.suptitle("Agent Count vs TF-IDF Sum for Keywords in Conversations")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("tfIdfSumsVsAgentCount.png")
print("Scatter plot of agent count vs TF-IDF sum is saved as tfIdfSumsVsAgentCount.png.")
