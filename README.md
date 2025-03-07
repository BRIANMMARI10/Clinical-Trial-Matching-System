# Clinical-Trial-Matching-System
Overview

This project is designed to match patients to eligible clinical trials using various NLP techniques, including traditional string matching, Word2Vec model training, and BERT-based similarity scoring. It extracts patient data, processes clinical trial eligibility criteria, and determines patient-trial matches based on similarity scores.

Setup Instructions

Prerequisites

Ensure you have the following installed:

Python 3.8+

pip package manager

Required Python libraries:

pip install pandas numpy torch transformers sentence-transformers scikit-learn openpyxl

Pre-trained BERT model (BioBERT or ClinicalBERT)

Installation

Clone the repository (if applicable):

git clone <repo-url>
cd clinical-trial-matching

Download the necessary patient and clinical trial datasets:

patients.csv and conditions.csv (Synthea data)

Clinical trial XML files (e.g., NCT00000135.xml)

Place these files in the project directory.

Usage

Step 1: Data Preprocessing

Run the following script to process patient data and merge conditions:

import pandas as pd
from datetime import datetime

def calculate_age(birthdate):
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

patient_data = pd.read_csv("patients.csv")
patient_data["AGE"] = patient_data["BIRTHDATE"].apply(calculate_age)
conditions = pd.read_csv("conditions.csv")
merged_data = patient_data.merge(conditions, left_on="Id", right_on="PATIENT", how="left")
merged_data.to_csv("merged_patient_data.csv", index=False)

Step 2: Extract Clinical Trial Eligibility Criteria

Run:

import xml.etree.ElementTree as ET
def extract_inclusion_criteria(trial_file):
    tree = ET.parse(trial_file)
    root = tree.getroot()
    text = root.find(".//eligibility/criteria/textblock")
    return text.text.split("\n") if text is not None else []

Step 3: Compute BERT-Based Similarity Scores

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy()

Step 4: Matching Patients to Trials

matched_patients = []
for _, patient in patients_df.iterrows():
    patient_embedding = get_bert_embedding(patient["DESCRIPTION"])
    for _, trial in trials_df.iterrows():
        trial_embedding = get_bert_embedding(trial["Eligibility_Criteria"])
        similarity_score = cosine_similarity(patient_embedding, trial_embedding)[0][0]
        if similarity_score > 0.6:
            matched_patients.append({
                "patientId": patient["Id"],
                "trialId": trial["Trial_ID"],
                "trialName": f"Trial {trial['Trial_ID']}",
                "eligibilityCriteriaMet": [f"{trial['Trial_ID']}[{similarity_score:.4f}]"]
            })

Step 5: Save Results

import json
import pandas as pd

output_df = pd.DataFrame(matched_patients)
output_df.to_excel("bert_matched_patients.xlsx", index=False)
with open("bert_matched_patients.json", "w") as json_file:
    json.dump(matched_patients, json_file, indent=4)

Expected Outputs

bert_matched_patients.xlsx: Excel file with patient-trial matches.

bert_matched_patients.json: JSON format:

{
  "patientId": "string",
  "eligibleTrials": [
    {
      "trialId": "string",
      "trialName": "string",
      "eligibilityCriteriaMet": ["string"]
    }
  ]
}

Notes

Modify similarity threshold (0.6 in script) based on precision-recall trade-off.

Use BioBERT for better medical terminology handling.

Contributors

Maintainer: [Your Name]

Contact: [Your Email]

License

This project is licensed under [MIT License].
