import re
import nltk
from nltk.stem import WordNetLemmatizer

jsonFilePaths = [
    # DRTAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup01/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/DRTAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/DRTAG-llm-selection.json",
    # IAAG llm selection
    "Novel-Approach/Responses/GPT-4o-backup01/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/IAAG-llm-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/IAAG-llm-selection.json",
    # DRTAG random selection
    "Novel-Approach/Responses/GPT-4o-backup01/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/DRTAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/DRTAG-random-selection.json",
    # IAAG random selection
    "Novel-Approach/Responses/GPT-4o-backup01/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/IAAG-random-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/IAAG-random-selection.json",
    # DRTAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup01/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/DRTAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/DRTAG-round-robin-selection.json",
    # IAAG round robin selection
    "Novel-Approach/Responses/GPT-4o-backup01/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup02/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup03/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup04/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup05/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup06/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup07/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup08/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup09/IAAG-round-robin-selection.json",
    "Novel-Approach/Responses/GPT-4o-backup10/IAAG-round-robin-selection.json",

    # autogen round robin selection
    "Existing-Solution/Responses/GPT-4o-backup01/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup02/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup03/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup04/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup05/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup06/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup07/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup08/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup09/autogen-round-robin-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup10/autogen-round-robin-selection.json",
    # autogen random selection
    "Existing-Solution/Responses/GPT-4o-backup01/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup02/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup03/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup04/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup05/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup06/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup07/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup08/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup09/autogen-random-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup10/autogen-random-selection.json",
    # autogen llm selection
    "Existing-Solution/Responses/GPT-4o-backup01/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup02/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup03/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup04/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup05/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup06/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup07/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup08/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup09/autogen-llm-selection.json",
    "Existing-Solution/Responses/GPT-4o-backup10/autogen-llm-selection.json",

    # # autogen llm selection (with only doctor and nurse)
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup01/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup02/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup03/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup04/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup05/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup06/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup07/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup08/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup09/autogen-llm-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup10/autogen-llm-selection.json",
    # # autogen random selection (with only doctor and nurse)
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup01/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup02/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup03/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup04/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup05/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup06/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup07/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup08/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup09/autogen-random-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup10/autogen-random-selection.json",
    # # autogen round robin selection (with only doctor and nurse)
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup01/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup02/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup03/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup04/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup05/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup06/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup07/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup08/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup09/autogen-round-robin-selection.json",
    # "Existing-Solution/Responses-No-Bias/GPT-4o-backup10/autogen-round-robin-selection.json",
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