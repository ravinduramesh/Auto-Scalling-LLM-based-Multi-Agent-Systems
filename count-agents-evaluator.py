import json

jsonFilePaths = [
    # autogen backup1
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-auto-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-random-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-round-robin-agent-selection.json",
    # autogen backup2
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-auto-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-random-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-round-robin-agent-selection.json",
    # autogen backup3
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-auto-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-random-agent-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-round-robin-agent-selection.json",
    # IAAG and DRTAG backup1
    "Novel-Approach/Responses/GPT-4o-backup1/dynamic-agent-creation-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/dynamic-agent-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/dynamic-agent-creation-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/initial-auto-creation-agent-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/initial-auto-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup1/initial-auto-creation-round-robin-selection.json",
    # IAAG and DRTAG backup2
    "Novel-Approach/Responses/GPT-4o-backup2/dynamic-agent-creation-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/dynamic-agent-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/dynamic-agent-creation-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/initial-auto-creation-agent-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/initial-auto-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/initial-auto-creation-round-robin-selection.json",
    # IAAG and DRTAG backup3
    "Novel-Approach/Responses/GPT-4o-backup3/dynamic-agent-creation-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/dynamic-agent-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/dynamic-agent-creation-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/initial-auto-creation-agent-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/initial-auto-creation-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/initial-auto-creation-round-robin-selection.json",
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

plt.figure(figsize=(10, 10))
plt.bar(agentCounts.keys(), agentCounts.values())
plt.xticks(rotation=90)
plt.ylabel("Number of agents")
plt.title("Number of agents in each conversation")
plt.tight_layout()
plt.savefig("agent-counts.png")
print("Graph is plotted successfully.")
