import json

import re
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

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
with open("cleaned-corpus.txt", "w") as file:
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
    "diet", "dietary", "hydrated", "hydration", "rest", "symptom diary", "stress", "breathing", "deepbreathing", "relaxation", "relax", "strenuous activity", "acupuncture", "allergy", "water", "diet", "heat", "fasting", "pain medication"
}
vectorizer = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2))
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

sums = tfidfTable.iloc[:, 1:].sum(axis=0)

plt.figure(figsize=(15, 10))
plt.bar(tfidfTable.columns[1:], sums)
plt.xticks(rotation=90)
plt.ylabel("TF-IDF Sum")
plt.title("TF-IDF Sum for each conversation")
plt.tight_layout()
plt.savefig("tf-idf-sums.png")

print("TF-IDF sums are saved as tf-idf-sums.png.")