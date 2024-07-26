import pandas as pd
import spacy
from spacy.util import minibatch, compounding
import random
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('ecommerce_dataset.csv')

# Data cleaning steps
df['product_description'] = df['product_description'].str.replace('[^a-zA-Z ]', '')
df['product_description'] = df['product_description'].str.lower()

# Splitting the dataset into training and testing sets
train, test = train_test_split(df, test_size=0.2)

# Load a blank English model
nlp = spacy.blank("en")

# Add the text categorizer to the pipeline
textcat = nlp.create_pipe("textcat")
nlp.add_pipe(textcat, last=True)

# Add labels to text categorizer
textcat.add_label("Places")
textcat.add_label("Things")
textcat.add_label("Activities")
textcat.add_label("People")

# Prepare training data
train_data = []
for row in train.itertuples():
    text = row.product_description
    category = row.product_category
    train_data.append((text, {"cats": {category: 1.0}}))

# Training the model
n_iter = 10
with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'textcat']):
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.5, losses=losses)
        print(f"Losses at iteration {i}: {losses}")

# Save the model
nlp.to_disk("text_classification_model")
