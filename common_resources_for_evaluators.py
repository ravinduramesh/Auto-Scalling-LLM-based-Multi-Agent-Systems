import re
import nltk
from nltk.stem import WordNetLemmatizer

jsonFilePaths = [
    # DRTAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/DRTAG-llm-selection.json",
    # IAAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/IAAG-llm-selection.json",
    # DRTAG random selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/DRTAG-random-selection.json",
    # IAAG random selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/IAAG-random-selection.json",
    # DRTAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup1/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/DRTAG-round-robin-selection.json",
    # IAAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup1/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup2/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup3/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup4/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup5/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup6/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup7/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup8/IAAG-round-robin-selection.json",

    # autogen round robin selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup4/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup5/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup6/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup7/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup8/autogen-round-robin-selection.json",
    # autogen random selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup4/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup5/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup6/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup7/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup8/autogen-random-selection.json",
    # autogen llm selection
    "Existing-Solution/Responses/GPT-4o-backup1/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup2/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup3/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup4/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup5/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup6/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup7/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup8/autogen-llm-selection.json",

#     # autogen llm selection (with only doctor and nurse)
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup1/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup2/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup3/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup4/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup5/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup6/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup7/autogen-llm-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup8/autogen-llm-selection.json",
#     # autogen random selection (with only doctor and nurse)
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup1/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup2/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup3/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup4/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup5/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup6/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup7/autogen-random-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup8/autogen-random-selection.json",
#     # autogen round robin selection (with only doctor and nurse)
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup1/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup2/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup3/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup4/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup5/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup6/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup7/autogen-round-robin-selection.json",
#     "Existing-Solution/Responses-No-Bias/GPT-4o-backup8/autogen-round-robin-selection.json"
]

stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])
    return text

ground_truth_vocab = [
    # Possible Illnesses
    "appendicitis", "gynecological", "kidney stone", "gastrointestinal", "colon cancer", "ileitis", "ovarian", "crohn disease", "colitis", "diverticulitis", "urinary tract", "musculoskeletal issue", "hernia",
    "cardiovascular", "gallbladder", "obstruction", "renal", "yersinia enterocolitica", "campylobacter jejuni", "ectopic pregnancy", "pelvic inflammatory", "endocrine disorder", "endometriosis", "inflammatory bowel",
    # Diagnostic Plans and Treatments
    "clinical examination", "blood test", "stool test", "ct", "urinalysis", "ultrasound", "surgery", "antibiotic", "pain management", "manage pain", "pain relief", "physical examination", "pysical exam", "medical history", "nephrology",
    "endoscopic evaluation", "laparoscopy", "laparoscopic", "allergies", "anesthetic", "anesthesia", "pelvic exam", "neurological examination", "hormone level", "mri", "probiotics", "urine",
    # Preventive Actions, Prior and Post Treatment Advice
    "diet", "dietary", "hydrated", "hydration", "rest", "symptom diary", "fever", "nausea", "vomiting", "bowel", "dizziness", "abdominal rigidity", "stress", "breathing", "deepbreathing", "relaxation", "relax", "strenuous activity", "acupuncture", "allergy", "water", "heat", "fasting", "pain medication"
]