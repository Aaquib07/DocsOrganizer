# Importing modules
import numpy as np
import pandas as pd
from bertopic import BERTopic

# Loading the dataset
df = pd.read_csv('data/tokyo_2020_tweets.csv')

# Excluding the rows from text column having float as the datatype
df = df.loc[df['text'].apply(type) != float]

# Extracting the text column
tweets = df['text'].to_list()

# Splitting the tweets into training and testing part
training_tweets = tweets[:160520]
testing_tweets = tweets[160520:]

# Initializing the BERTopic model
model = BERTopic(verbose=True)

# Training the model and making predictions on training tweets
topics, probabilities = model.fit_transform(training_tweets)

# Count of each topics extracted
# Topic -1 is the miscellaneous topic (outliers topic)
freq_of_topics = model.get_topic_freq()

# Making predictions on the testing tweets
topics, probs = model.transform(testing_tweets)

# Visualizing the prediction as a dataframe
predictions_df = pd.DataFrame({
    'Predicted Topics': topics,
    'Text': testing_tweets,
})

# Saving the model
model.save('BERTopic_model')
