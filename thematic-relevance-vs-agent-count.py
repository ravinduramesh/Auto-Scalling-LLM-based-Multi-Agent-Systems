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

def count_agents_in_json(jsonFilePath):
    with open(jsonFilePath) as jsonFile:
        jsonData = json.load(jsonFile)
    
    # Count number of unique agents (excluding the 'user' role)
    agents = set()
    for entry in jsonData:
        agents.add(entry["role"])
    
    # Subtract 1 to exclude 'user' role from agent count
    return len(agents) - 1

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
vocab_string = ". ".join(ground_truth_vocab)

agentCountsAndThematicRelevance = dict()

for jsonFilePath in jsonFilePaths:
    text = get_text_from_json(jsonFilePath)
    cleaned_text = clean_text(text)
    
    # Encode all utterances and the vocab string
    utterance_embeddings = bert_model.encode([cleaned_text], convert_to_numpy=True, show_progress_bar=False)
    vocab_embedding = bert_model.encode([vocab_string], convert_to_numpy=True, show_progress_bar=False)[0]
    # Calculate cosine similarity between each utterance and the vocab embedding
    similarities = cosine_similarity(utterance_embeddings, vocab_embedding.reshape(1, -1)).flatten()
    thematic_relevance = float(np.mean(similarities))
    
    # Count agents
    agent_count = count_agents_in_json(jsonFilePath)
    
    # Store agent count and thematic relevance
    filename = jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]
    agentCountsAndThematicRelevance[filename] = [agent_count, thematic_relevance]

print("Agent counts and thematic relevance scores calculated for each JSON file.")

# Plot scatter plots for each selection type
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
    for key, value in agentCountsAndThematicRelevance.items():
        if selection_type in key:
            label = (key.split("\n")[-1]).split("-")[0]
            color = scatterColors[label]
            ax.scatter(value[0], value[1], color=color, s=100)

    ax.set_title(selection_titles[idx])
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Thematic Relevance")
    ax.grid(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
fig.legend(handles=legend_elements, loc="upper right", title="LLM-based MAS approach")
fig.suptitle("Agent Count vs Thematic Relevance to Ground Truth Vocabulary")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("thematicRelevanceVsAgentCount.png")
print("Scatter plot of agent count vs thematic relevance is saved as thematicRelevanceVsAgentCount.png.")
