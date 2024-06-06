import json

jsonFilePaths = [
    # autogen backup1
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-auto-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-round-robin-selection.json",
    # autogen backup2
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-auto-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-round-robin-selection.json",
    # autogen backup3
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-auto-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-round-robin-selection.json",
    # IAAG and DRTAG backup1
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-round-robin-selection.json",
    # IAAG and DRTAG backup2
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-round-robin-selection.json",
    # IAAG and DRTAG backup3
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-round-robin-selection.json",
]

agentCounts = dict()

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Count number of agents
    agents = set()
    for entry in jsonData:
        agents.add(entry["role"])

    print(agents)
    filename = jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]
    agentCounts[filename] = len(agents) - 1 # Exclude the patient

print("Number of agents are counted for each JSON file.")

# Plot a graph of agent counts
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 15})
barColors = []
for label in agentCounts.keys():
    if label.split("\n")[1].startswith("autogen"):
        barColors.append('red')
    elif label.split("\n")[1].startswith("DRTAG"):
        barColors.append('green')
    else:
        barColors.append('blue')

plt.bar(agentCounts.keys(), agentCounts.values(), color=barColors)
plt.xticks(rotation=90)
plt.ylabel("Number of agents")
plt.title("Number of agents in each conversation")
plt.tight_layout()
plt.savefig("agentCounts.png")
print("Graph is plotted successfully.")
