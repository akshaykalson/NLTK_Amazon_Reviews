import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

df = pd.read_csv('resources/Reviews.csv')

# print(df.head())
print(df.shape)
df = df.head(10000) #in case we need to downsample the dataset for our processor

plot1 = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
# Show the plot
plt.xlabel('Review Score')
plt.ylabel('Number of Reviews')
plt.show()

#basic NLTK

example = df['Text'][50]
print(example)

#now we will see what NLTK can do
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
tokens = nltk.word_tokenize(example)
print(tokens)

#using NLTK to do part of speech tagging

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()



