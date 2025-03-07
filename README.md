# Clinical-Trial-Matching-System
Description
This project implements a system for matching patients to eligible clinical trials using NLP techniques. It processes patient data and clinical trial criteria to determine suitable matches based on similarity scores.
Features
Patient data preprocessing and merging
Clinical trial eligibility criteria extraction
BERT-based similarity scoring
Patient-trial matching algorithm
Installation
bash
git clone <repo-url>
cd clinical-trial-matching
pip install -r requirements.txt

Usage
Preprocess patient data:
python
python preprocess_data.py
Extract clinical trial criteria:
python
python extract_criteria.py
Compute similarity scores and match patients:
python
python match_patients.py
View results in bert_matched_patients.xlsx and bert_matched_patients.json
Requirements
Python 3.8+
pandas
numpy
torch
transformers
sentence-transformers
scikit-learn
openpyxl
Configuration
Adjust the similarity threshold in match_patients.py to fine-tune matching precision.
Data
Place the following files in the project directory:
patients.csv
conditions.csv
Clinical trial XML files (e.g., NCT00000135.xml)
Output
bert_matched_patients.xlsx: Excel file with patient-trial matches
bert_matched_patients.json: JSON file with detailed match information
Contributing
Contributions are welcome. Please open an issue or submit a pull request.
License
This project is licensed under the MIT License.
