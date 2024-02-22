#!/usr/bin/env python
# coding: utf-8

# # An Analysis of Ghanaian Gospel Choral Songs

# By Veronica Annor

# ### Background

# The Ghanaian choral music dataset represents a vibrant collection deeply rooted in the rich cultural heritage of Ghana. This corpus encompasses a diverse array of choral compositions, showcasing the multifaceted melodic intricacies, rhythmic vibrancy, and soul-stirring harmonies inherent in Ghanaian musical traditions. These gospel compositions often narrate tales of biblical events, thanksgiving, solemnity,and communal celebrations, embodying the essence of faith, unity and expression through music within Ghana's cultural tapestry.

# ### Import libraries 

# In[11]:


import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


# ## 1. Data Collection and Preprocessing

# ### Load and Preprocess Audio Data

# In[2]:


# Function to extract audio features
def extract_features(file_path):
    # Load audio file
    audio_data, sampling_rate = librosa.load(file_path, sr=None)

    return audio_data, sampling_rate

# Folder path containing audio files
folder_path = '/Users/veronicaannor/Documents/Personal projects/songs_mp3'

# List to store audio features
all_features = []

# Process each audio file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.mp3'):
        file_path = os.path.join(folder_path, file)
        
        # Extract features
        audio_data, sampling_rate = extract_features(file_path)
        
        # Store extracted features in a dictionary
        features = {
            'file_name': file,
            'audio_data': audio_data,
            'sampling_rate': sampling_rate
        }
        
        # Append features to the list
        all_features.append(features)


# ### Extract Spectrogram Features

# In[3]:


def extract_spectrogram(audio_data, sampling_rate):
    spectrogram = np.abs(librosa.stft(audio_data))
    return spectrogram

# Extract spectrogram features for each audio file
for features in all_features:
    features['spectrogram'] = extract_spectrogram(features['audio_data'], features['sampling_rate'])


# ### Extract MFCCs

# In[4]:


def extract_mfcc(audio_data, sampling_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)
    return mfccs

# Extract MFCCs for each audio file
for features in all_features:
    features['mfcc'] = extract_mfcc(features['audio_data'], features['sampling_rate'])


# ### Extract Tempo, Beat, Rhythm

# In[5]:


def extract_tempo_beat(audio_data, sampling_rate):
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sampling_rate)
    return tempo, beat_frames

# Extract tempo and beat for each audio file
for features in all_features:
    features['tempo'], features['beat_frames'] = extract_tempo_beat(features['audio_data'], features['sampling_rate'])


# ### Extract Harmonic and Percussive Components

# In[6]:


def extract_harmonic_percussive(audio_data):
    harmonic, percussive = librosa.effects.hpss(audio_data)
    return harmonic, percussive

# Extract harmonic and percussive components for each audio file
for features in all_features:
    features['harmonic'], features['percussive'] = extract_harmonic_percussive(features['audio_data'])


# ## 2. Exploratory Data Analysis (EDA)

# ### Exploring Feature Distributions Across Songs

# In[12]:


# Extracting tempo values from features
tempo_values = [features['tempo'] for features in all_features]

# Histogram of tempo values across songs
plt.figure(figsize=(8, 6))
plt.hist(tempo_values, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Tempo')
plt.ylabel('Frequency')
plt.title('Distribution of Tempo Across Songs')
plt.show()

# Box plot of tempo values across songs
plt.figure(figsize=(8, 6))
plt.boxplot(tempo_values, vert=False)
plt.xlabel('Tempo')
plt.title('Distribution of Tempo Across Songs')
plt.show()


# ### Explanation of Tempo Distribution

# Bar Chart: The bar chart showcases how different tempos are represented across the songs. Each bar represents a specific tempo range, and the height of the bar indicates the number of songs characterised by that tempo. For instance, taller bars illustrate a higher number of songs with a particular tempo range, while shorter bars represent fewer songs in that tempo range.

# Box Plot: The box plot offers a visual summary of the tempo data distribution. It presents the range of tempos observed within the dataset, displaying the median (middle line), interquartile range (box), and potential outliers (data points beyond the whiskers). This visualisation helps us understand the central tendencies and variability in tempo among the songs.

# ## Unsupervised Clustering for Emotion Recognition (Without Explicit Labels)

# ### K-means Clustering on Extracted Features

# In[14]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[19]:


# Extract relevant features for clustering (e.g., MFCCs, tempo, etc.)
# Create a feature matrix containing the selected features for each song
feature_matrix = []

for features in all_features:
    # Example: Using MFCCs and Tempo for clustering
    mfcc_mean = np.mean(features['mfcc'], axis=1)  # Taking mean of MFCCs as a representative feature
    tempo = features['tempo']
    # Add more features as needed
    
    # Creating a feature vector for each song
    song_features = np.concatenate([mfcc_mean, [tempo]])  # Concatenating features into a single vector
    feature_matrix.append(song_features)

# Standardize the feature matrix (important for K-means)
scaler = StandardScaler()
scaled_feature_matrix = scaler.fit_transform(feature_matrix)

# Apply K-means clustering with explicit n_init value
num_clusters = 5  # Define the number of clusters (can be adjusted)
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Set n_init explicitly
cluster_labels = kmeans.fit_predict(scaled_feature_matrix)


# Assign cluster labels to each song
for idx, features in enumerate(all_features):
    features['cluster_label'] = cluster_labels[idx]


# ### Printing cluster labels for each song

# In[20]:


for idx, features in enumerate(all_features):
    print(f"Song: {features['file_name']} - Cluster Label: {features['cluster_label']}")

# Count the number of songs in each cluster
from collections import Counter
cluster_counts = Counter(features['cluster_label'] for features in all_features)
print("\nNumber of songs in each cluster:")
print(cluster_counts)


# ### Creating a Table for Cluster Labels

# In[21]:


import pandas as pd

# Create a DataFrame for cluster labels
cluster_data = {
    'Song': [features['file_name'] for features in all_features],
    'Cluster Label': [features['cluster_label'] for features in all_features]
}

cluster_df = pd.DataFrame(cluster_data)
print(cluster_df)


# ### Visualising Cluster Distribution

# In[22]:


import matplotlib.pyplot as plt

# Plotting cluster distribution
plt.figure(figsize=(8, 6))
plt.bar(cluster_counts.keys(), cluster_counts.values())
plt.xlabel('Cluster Label')
plt.ylabel('Number of Songs')
plt.title('Cluster Distribution')
plt.xticks(list(cluster_counts.keys()))
plt.show()


# ### Conclusion of Clusters:

# The exploration of the Ghanaian choral music dataset using data-driven techniques has provided intriguing insights into the inherent patterns within these musical compositions. Employing unsupervised learning algorithms like K-means clustering on extracted audio features, a nuanced understanding of the musical pieces' structural similarities and variations has emerged.
# 
# The clustering analysis revealed distinct clusters, showcasing how songs within each cluster share commonalities in their audio features. This grouping offers an insightful perspective, potentially hinting at thematic or stylistic similarities present in the dataset. Exploring these clusters aids in unveiling the underlying nuances and diverse elements that contribute to the rich tapestry of Ghanaian choral music, offering a lens into its multifaceted nature.
