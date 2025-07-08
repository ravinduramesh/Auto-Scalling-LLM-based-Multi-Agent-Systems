import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from lexical_diversity import lex_div as ld
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

def get_text_from_json(jsonFilePath):
    """Extract all text content from a JSON file."""
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)
    all_text = " ".join(entry["content"] for entry in jsonData)
    return all_text

def count_agents(jsonFilePath):
    """Count the number of unique agents in a JSON file."""
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)
    
    agents = set()
    for entry in jsonData:
        agents.add(entry["role"])
    
    return len(agents) - 1  # Subtract 1 to exclude system/moderator

def calculate_mtld_score(text):
    """Calculate MTLD (Measure of Textual Lexical Diversity) score."""
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    return ld.mtld(tokens)

def calculate_thematic_relevance(text, vocab_embedding, bert_model):
    """Calculate thematic relevance using BERT embeddings."""
    cleaned_text = clean_text(text)
    
    # Encode the text
    text_embedding = bert_model.encode([cleaned_text], convert_to_numpy=True, show_progress_bar=False)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(text_embedding, vocab_embedding.reshape(1, -1)).flatten()
    return float(np.mean(similarity))

def get_approach_info(filename):
    """Extract approach type and selection method from filename."""
    if "autogen" in filename:
        approach = "Autogen"
        color = 'orangered'
        marker = 'o'
    elif "DRTAG" in filename:
        approach = "DRTAG"
        color = 'lawngreen'
        marker = 's'
    elif "IAAG" in filename:
        approach = "IAAG"
        color = 'dodgerblue'
        marker = '^'
    else:
        approach = "Unknown"
        color = 'gray'
        marker = 'x'
    
    if "llm-selection" in filename:
        selection = "LLM Selection"
    elif "random-selection" in filename:
        selection = "Random Selection"
    elif "round-robin" in filename:
        selection = "Round Robin"
    else:
        selection = "Unknown"
    
    return approach, selection, color, marker

# Initialize BERT model and vocabulary embedding
print("Loading BERT model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
vocab_string = ". ".join(ground_truth_vocab)
vocab_embedding = bert_model.encode([vocab_string], convert_to_numpy=True, show_progress_bar=False)[0]

# Store all data
data_points = []

print("Processing JSON files...")
for jsonFilePath in jsonFilePaths:
    print(f"Processing: {jsonFilePath}")
    
    # Get text content
    text = get_text_from_json(jsonFilePath)
    
    # Calculate metrics
    agent_count = count_agents(jsonFilePath)
    mtld_score = calculate_mtld_score(text)
    thematic_relevance = calculate_thematic_relevance(text, vocab_embedding, bert_model)
    
    # Get approach and selection info
    filename = jsonFilePath.split("/")[-2] + '\n' + jsonFilePath.split("/")[-1]
    approach, selection, color, marker = get_approach_info(filename)
    
    data_points.append({
        'filename': filename,
        'agent_count': agent_count,
        'mtld_score': mtld_score,
        'thematic_relevance': thematic_relevance,
        'approach': approach,
        'selection': selection,
        'color': color,
        'marker': marker
    })

print("Creating 3D scatter plot...")

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')

# Define colors and markers for different approaches
approach_colors = {'Autogen': 'orangered', 'DRTAG': 'lawngreen', 'IAAG': 'dodgerblue'}
approach_markers = {'Autogen': 'o', 'DRTAG': 's', 'IAAG': '^'}

# Plot data points for each approach
for approach in ['Autogen', 'DRTAG', 'IAAG']:
    # Filter data for current approach
    approach_data = [dp for dp in data_points if dp['approach'] == approach]
    
    if approach_data:  # Check if there's data for this approach
        xs = [dp['agent_count'] for dp in approach_data]
        ys = [dp['mtld_score'] for dp in approach_data]
        zs = [dp['thematic_relevance'] for dp in approach_data]
        
        ax.scatter(xs, ys, zs, 
                  c=approach_colors[approach], 
                  marker=approach_markers[approach],
                  s=100, alpha=0.7, label=approach)

ax.set_xlabel('Agent Count')
ax.set_ylabel('MTLD Score')
ax.set_zlabel('Thematic Relevance')
ax.set_title('3D Scatter Plot: Agent Count vs MTLD Score vs Thematic Relevance')
ax.legend()

plt.savefig("mtld-vs-thematic-relevance-vs-agent-count-3d.png", dpi=300, bbox_inches='tight')
print("3D scatter plot saved as 'mtld-vs-thematic-relevance-vs-agent-count-3d.png'")

plt.show()
print("\nScript completed successfully!")
