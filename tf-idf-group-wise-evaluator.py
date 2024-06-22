import json

import re
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
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

corpus = []
autogenAutoCorpus = []
autogenRandomCorpus = []
autogenRoundRobinCorpus = []
drtagLlmCorpus = []
drtagRandomCorpus = []
drtagRoundRobinCorpus = []
iaagLlmCorpus = []
iaagRandomCorpus = []
iaagRoundRobinCorpus = []

# Read data from the JSON file
for jsonFilePath in jsonFilePaths:
    with open(jsonFilePath, "r") as jsonFile:
        jsonData = json.load(jsonFile)

    # Convert data to a plain text format (Agent - Dialog \n Agent - Dialog)
    plainTextData = ""
    for entry in jsonData:
        plainTextData += f"{entry['role']} - {entry['content']}\n"

    if "autogen-llm-selection" in jsonFilePath:
        autogenAutoCorpus.append(plainTextData)
    elif "autogen-random-selection" in jsonFilePath:
        autogenRandomCorpus.append(plainTextData)
    elif "autogen-round-robin-selection" in jsonFilePath:
        autogenRoundRobinCorpus.append(plainTextData)
    elif "DRTAG-llm-selection" in jsonFilePath:
        drtagLlmCorpus.append(plainTextData)
    elif "DRTAG-random-selection" in jsonFilePath:
        drtagRandomCorpus.append(plainTextData)
    elif "DRTAG-round-robin-selection" in jsonFilePath:
        drtagRoundRobinCorpus.append(plainTextData)
    elif "IAAG-llm-selection" in jsonFilePath:
        iaagLlmCorpus.append(plainTextData)
    elif "IAAG-random-selection" in jsonFilePath:
        iaagRandomCorpus.append(plainTextData)
    elif "IAAG-round-robin-selection" in jsonFilePath:
        iaagRoundRobinCorpus.append(plainTextData)

corpus.append('\n '.join(autogenAutoCorpus))
corpus.append('\n '.join(autogenRandomCorpus))
corpus.append('\n '.join(autogenRoundRobinCorpus))
corpus.append('\n '.join(drtagLlmCorpus))
corpus.append('\n '.join(drtagRandomCorpus))
corpus.append('\n '.join(drtagRoundRobinCorpus))
corpus.append('\n '.join(iaagLlmCorpus))
corpus.append('\n '.join(iaagRandomCorpus))
corpus.append('\n '.join(iaagRoundRobinCorpus))

print("Corpus with plain text documents is created.")

# Clean the text data and remove stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])
    return text

cleanedCorpus = [clean_text(doc) for doc in corpus]
# save cleaned corpus to a txt file
with open("cleaned-corpus-group-wise.txt", "w") as file:
    for doc in cleanedCorpus:
        file.write(doc + "\n")
print("Text data is cleaned and stopwords are removed.")

# Create the TF-IDF vectorizer
vocab = {
    # Possible Illnesses
    "appendicitis", "gynecological", "kidney stone", "gastrointestinal", "colon cancer", "ileitis", "ovarian", "crohn disease", "colitis", "diverticulitis", "urinary tract", "musculoskeletal issue", "hernia",
    "cardiovascular", "gallbladder", "obstruction", "renal", "yersinia enterocolitica", "campylobacter jejuni", "ectopic pregnancy", "pelvic inflammatory", "endocrine disorder", "endometriosis", "inflammatory bowel",
    # Diagnostic Plans and Treatments 
    "clinical examination", "blood test", "stool test", "ct", "urinalysis", "ultrasound", "surgery", "antibiotic", "pain management", "manage pain", "pain relief", "physical examination", "pysical exam", "medical history", "nephrology",
    "endoscopic evaluation", "laparoscopy", "laparoscopic", "allergies", "anesthetic", "anesthesia", "pelvic exam", "neurological examination", "hormone level", "mri", "endoscopic evaluation", "probiotics", "urine",
    # Preventive Actions, Prior and Post Treatment Advice  
    "diet", "dietary", "hydrated", "hydration", "rest", "symptom diary", "fever", "nausea", "vomiting", "bowel", "dizziness", "abdominal rigidity", "stress", "breathing", "deepbreathing", "relaxation", "relax", "strenuous activity", "acupuncture", "allergy", "water", "diet", "heat", "fasting", "pain medication"
}
vectorizer = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2))
tfidfMatrix = vectorizer.fit_transform(cleanedCorpus)

# Create tfidf table to output
outputVocabulary = vectorizer.get_feature_names_out()
documentNames = ["autogen-llm-selection", "autogen-random-selection", "autogen-round-robin-selection", "DRTAG-llm-selection", "DRTAG-random-selection", "DRTAG-round-robin-selection", "IAAG-llm-selection", "IAAG-random-selection", "IAAG-round-robin-selection"]

tfidfArray = tfidfMatrix.toarray()
transposedTfidfArray = tfidfArray.T
tfidfTable = pd.DataFrame(transposedTfidfArray, columns=documentNames, index=outputVocabulary)

tfidfTable.to_csv("tfidf-table-group-wise.csv")
print("TF-IDF table is created and saved as tfidf-table.csv.")

sums = tfidfTable.iloc[:, 0:].sum(axis=0)

plt.figure(figsize=(20, 10))
plt.rcParams.update({'font.size': 15})
barColors = []
for label in tfidfTable.columns[0:]:
    if label.startswith("autogen"):
        barColors.append('orangered')
    elif label.startswith("DRTAG"):
        barColors.append('lawngreen')
    else:
        barColors.append('dodgerblue')

plt.bar(tfidfTable.columns[0:], sums, color=barColors)
plt.xticks(rotation=90)
plt.ylabel("TF-IDF Sums")
plt.title("Summations of all keywords' TF-IDF values within each grouped conversation by their generated approach")
plt.tight_layout()
plt.savefig("tfidfGroupWise.png")

print("TF-IDF sums are saved as tf-idf-group-wise.png")