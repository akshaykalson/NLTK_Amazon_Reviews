import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load your dataset (assuming 'Reviews.csv' is in the 'resources' folder)
df = pd.read_csv('resources/Reviews.csv')
df = df.head(500)

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Calculate polarity scores for a sample sentence
print(sia.polarity_scores('The product that I bought was not good, I mean I have seen better products. I would not mind if they increased the quality'))

# Initialize an empty dictionary to store results
res = {}

# Iterate over each row in the dataframe and calculate polarity scores
for i, row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Convert the dictionary 'res' to a pandas DataFrame 'vaders'
vaders = pd.DataFrame(res).T

# Merge 'vaders' with 'df' based on 'Id'
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left', on='Id')

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row and 3 columns

# Plotting positive, neutral, and negative scores
sns.barplot(data=vaders, x='Score', y='pos', ax=axes[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axes[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axes[2])

axes[0].set_title('Positive Scores')
axes[1].set_title('Neutral Scores')
axes[2].set_title('Negative Scores')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
