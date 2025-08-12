import json
import matplotlib.pyplot as plt
from lexical_diversity import lex_div
import numpy as np

from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

def calculate_binary_weight_score(jsonFilePath):
    """Calculate binary weight score for task-related content coverage by newly created agents"""
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
    
    ground_truth_vocab_set = set(ground_truth_vocab)
    termsToScoreConversation = ground_truth_vocab_set.intersection(newlyGeneratedAgentsTermSet - initialAgentsTermSet)

    return len(termsToScoreConversation)

mtldAndBinaryWeight = dict()

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Calculate MTLD score
    all_text = " ".join(entry["content"] for entry in jsonData)
    cleaned_text = clean_text(all_text)
    
    # Calculate MTLD (Measure of Textual Lexical Diversity)
    mtld_score = lex_div.mtld(cleaned_text.split())
    
    # Calculate binary weight score for task-related content coverage
    binary_weight_score = calculate_binary_weight_score(jsonFilePath)

    filename = 'conv' + jsonFilePath.split("/")[-2][-2:] + '-' + jsonFilePath.split("/")[-1]
    mtldAndBinaryWeight[filename] = [mtld_score, binary_weight_score]

print("MTLD scores and binary weight scores are calculated for each JSON file.")

# Plot a scatter plot of MTLD score vs binary weight score
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
    for key, value in mtldAndBinaryWeight.items():
        if selection_type in key:
            label = key.split("-")[1]
            color = scatterColors[label]
            ax.scatter(value[0], value[1], color=color, s=100)

    ax.set_title(selection_titles[idx])
    ax.set_xlabel("MTLD Score")
    ax.set_ylabel("Binary Weight Score (Task-Related Content)")
    ax.grid(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', label='Autogen'),
    Patch(facecolor='lawngreen', label='DRTAG'),
    Patch(facecolor='dodgerblue', label='IAAG')
]
fig.legend(handles=legend_elements, loc="upper right", title="LLM-based MAS approach")
fig.suptitle("MTLD Score vs Binary Weight Score (Task-Related Content Coverage)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mtldVsBinaryWeightForTaskRelatedContent.png")
print("Scatter plot of MTLD score vs binary weight score is saved as mtldVsBinaryWeightForTaskRelatedContent.png")

# Measure the correlation between MTLD score and binary weight score
mtld_scores = np.array([value[0] for value in mtldAndBinaryWeight.values()])
binary_weight_scores = np.array([value[1] for value in mtldAndBinaryWeight.values()])

correlation = np.corrcoef(mtld_scores, binary_weight_scores)[0, 1]
print(f"Correlation between MTLD score and binary weight score: {correlation:.2f}")
if abs(correlation) > 0.5:
    conclusion = "There is a strong correlation between MTLD score and binary weight score for task-related content coverage."
else:
    conclusion = "There is a weak correlation between MTLD score and binary weight score for task-related content coverage."

# Save the conclusion to a text file
with open("mtld_vs_binary_weight_for_task_related_content_conclusion.txt", "w") as conclusion_file:
    conclusion_file.write(f"Correlation between MTLD score and binary weight score for task-related content coverage: {correlation:.2f}\n")
    conclusion_file.write(conclusion)

print("Analysis complete. Results saved to file.")
