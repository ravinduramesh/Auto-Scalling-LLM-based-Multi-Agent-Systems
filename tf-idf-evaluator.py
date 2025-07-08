import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

from common_resources_for_evaluators import jsonFilePaths, clean_text, ground_truth_vocab

corpus = []

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Convert data to a plain text format (Agent - Dialog \n Agent - Dialog)
    plainTextData = ""
    for entry in jsonData:
        plainTextData += f"{entry['role']} - {entry['content']}\n"

    corpus.append(plainTextData)

print("Corpus with plain text documents is created.")

cleanedCorpus = [clean_text(doc) for doc in corpus]
# save cleaned corpus to a txt file
with open("cleaned-corpus.txt", "w") as file:
    for doc in cleanedCorpus:
        file.write(doc + "\n")
print("Text data is cleaned and stopwords are removed.")

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(vocabulary=ground_truth_vocab, ngram_range=(1, 2))
tfidfMatrix = vectorizer.fit_transform(cleanedCorpus)

# Create tfidf table to output
outputVocabulary = vectorizer.get_feature_names_out()
documentNames = []
for path in jsonFilePaths:
    splitPath = path.split('/')
    documentNames.append(splitPath[-2]+'/'+splitPath[-1])

tfidfArray = tfidfMatrix.toarray()
transposedTfidfArray = tfidfArray.T
tfidfTable = pd.DataFrame(transposedTfidfArray, columns=documentNames, index=outputVocabulary)

tfidfTable.to_csv("tfidf-table.csv")
print("TF-IDF table is created and saved as tfidf-table.csv.")

sums = tfidfTable.iloc[:, 0:].sum(axis=0)

# convert '/' in column in tfidfTable to '\n' for better readability
tfidfTable.columns = [col.replace('/', '\n') for col in tfidfTable.columns]

# Create colors and labels for each bar
barColors = []
labels = []
for label in tfidfTable.columns[0:]:
    if label.split("\n")[1].startswith("autogen"):
        barColors.append('orangered')
    elif label.split("\n")[1].startswith("DRTAG"):
        barColors.append('lawngreen')
    else:
        barColors.append('dodgerblue')

# Sort the data by TF-IDF sums in descending order
sorted_indices = sums.argsort()[::-1]
sorted_columns = tfidfTable.columns[sorted_indices]
sorted_sums = sums[sorted_indices]
sorted_colors = [barColors[i] for i in sorted_indices]

plt.figure(figsize=(35, 12))
plt.rcParams.update({'font.size': 15})

bars = plt.bar(sorted_columns, sorted_sums, color=sorted_colors)
plt.xticks(rotation=90)
plt.ylabel("TF-IDF Sums")
plt.title("Summations of all keywords' TF-IDF values within each conversation")

# Create legend
legend_handles = []
legend_labels = []
color_map = {'Autogen': 'orangered', 'DRTAG': 'lawngreen', 'IAAG': 'dodgerblue'}
for label in ['Autogen', 'DRTAG', 'IAAG']:
    if label in labels:
        legend_handles.append(plt.Rectangle((0,0),1,1, color=color_map[label]))
        legend_labels.append(label)

plt.legend(legend_handles, legend_labels, loc='upper right')
plt.tight_layout()
plt.savefig("tfidfSumsOfConversations.png")
print("TF-IDF sums are saved as tfIdfSums.png.")


# Statistical analysis with Mann-Whitney U rank test on tf-idf summasion scores
import numpy as np
from scipy.stats import mannwhitneyu
import scikit_posthocs as sp

standardSignificanceLevel = 0.05
conclusions = []

# Group tf-idf summation scores by label
# Create a dictionary mapping document names to their TF-IDF sums
tfidf_score_sums = {}
for i, doc_name in enumerate(documentNames):
    tfidf_score_sums[doc_name] = sums.iloc[i]

autogen_scores = [score for label, score in tfidf_score_sums.items() if "autogen" in label]
drtag_scores = [score for label, score in tfidf_score_sums.items() if "DRTAG" in label]
iaag_scores = [score for label, score in tfidf_score_sums.items() if "IAAG" in label]

# autogen_llm_selection_scores = [score for label, score in tfidf_score_sums.items() if "autogen-llm-selection" in label]
# drtag_llm_selection_scores = [score for label, score in tfidf_score_sums.items() if "DRTAG-llm-selection" in label]
# iaag_llm_selection_scores = [score for label, score in tfidf_score_sums.items() if "IAAG-llm-selection" in label]
# autogen_random_selection_scores = [score for label, score in tfidf_score_sums.items() if "autogen-random-selection" in label]
# drtag_random_selection_scores = [score for label, score in tfidf_score_sums.items() if "DRTAG-random-selection" in label]
# iaag_random_selection_scores = [score for label, score in tfidf_score_sums.items() if "IAAG-random-selection" in label]
# autogen_round_robin_selection_scores = [score for label, score in tfidf_score_sums.items() if "autogen-round-robin" in label]
# drtag_round_robin_selection_scores = [score for label, score in tfidf_score_sums.items() if "DRTAG-round-robin" in label]
# iaag_round_robin_selection_scores = [score for label, score in tfidf_score_sums.items() if "IAAG-round-robin" in label]

# Mann-Whitney U rank test to check if DRTAG is better than Autogen
stat, p = mannwhitneyu(drtag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's TF-IDF scores are better than Autogen's TF-IDF scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relevant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relevant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if IAAG is better than Autogen
stat, p = mannwhitneyu(iaag_scores, autogen_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (IAAG's TF-IDF scores are better than Autogen's TF-IDF scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using IAAG contains more keywords relevant to the scenario than discussions generated using Autogen.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using IAAG contains more keywords relevant to the scenario than discussions generated using Autogen.")
conclusions.append("")

# Mann-Whitney U rank test to check if DRTAG is better than IAAG
stat, p = mannwhitneyu(drtag_scores, iaag_scores, alternative='greater')
conclusions.append(f"Mann-Whitney U Test (DRTAG's TF-IDF scores are better than IAAG's TF-IDF scores): H={stat:.3f}, p={p:.4f}")
if p < standardSignificanceLevel:
    conclusions.append("Conclusion: We reject the null hypothesis. There is statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relevant to the scenario than discussions generated using IAAG.")
else:
    conclusions.append("Conclusion: We fail to reject the null hypothesis. There is no statistically significant evidence to conclude that discussions generated using DRTAG contains more keywords relevant to the scenario than discussions generated using IAAG.")
conclusions.append("")
