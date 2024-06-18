import json
import pandas as pd
import matplotlib.pyplot as plt

jsonFilePaths = [
    # autogen llm selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-llm-selection.json",
    # DRTAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-llm-selection.json",
    # IAAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-llm-selection.json",
    # autogen random selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-random-selection.json",
    # DRTAG random selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-random-selection.json",
    # IAAG random selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-random-selection.json",
    # autogen round robin selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-round-robin-selection.json",
    # DRTAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-round-robin-selection.json",
    # IAAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-round-robin-selection.json",
]

agentCountsAndTfIdf = dict()

# Get tfidf values for each document
tfidfTable = pd.read_csv('tfidf-table.csv')
tfidfSums = dict()
for column in tfidfTable.columns[1:]:
    tfidfSums[column] = tfidfTable[column].sum()

print(tfidfSums)

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
print(agentCountsAndTfIdf)

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

fig.legend(loc="upper right", title="LLM-based MAS approach", labels=["Autogen", "DRTAG", "IAAG"])
fig.suptitle("Agent Count vs Binary Weight for New Knowledge")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("agentCountVsTfIdfSums.png")
print("Scatter plot of agent count vs binary weight is saved as agentCountVsNewKnowledgeBinaryWeight3.png.")
