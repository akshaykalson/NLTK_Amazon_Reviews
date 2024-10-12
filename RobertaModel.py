# Import required libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from scipy.special import softmax
import torch

# Load your dataset (assuming 'Reviews.csv' is in the 'resources' folder)
df = pd.read_csv('resources/Reviews.csv')
df = df.head(100)

# Example text from the dataset
example = df['Text'][50]

# Load the pre-trained model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define a function to calculate polarity scores using Roberta
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')

    # Get model predictions
    with torch.no_grad():  # Disable gradient calculation
        output = model(**encoded_text)

    # Extract scores and apply softmax to get probabilities
    scores = output.logits[0].numpy()
    scores = softmax(scores)

    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Initialize a dictionary to store results
res = {}
for i, row in df.iterrows():
    try:
        text = row['Text']
        myid = row['Id']
        roberta_result = polarity_scores_roberta(text)
        res[myid] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}')

# Convert the results dictionary to a DataFrame
roberta = pd.DataFrame(res).T

# Merge 'roberta' with 'df' based on 'Id'
roberta = roberta.reset_index().rename(columns={'index': 'Id'})
roberta = roberta.merge(df, how='left', on='Id')

# Print the DataFrame
print(roberta.head())

# Store the results of 'roberta' DataFrame into an Excel file
output_excel_file = 'roberta_sentiment_scores.xlsx'
roberta.to_excel(output_excel_file, index=False)

print(f"DataFrame saved to {output_excel_file}")
