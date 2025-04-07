#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\basanth\Downloads\Netflix\netflix_movies_detailed_up_to_2025.csv')


# In[3]:


df


# In[7]:


df.head()


# In[8]:


df['director']


# In[9]:


df.shape


# In[11]:


# Display basic information
df.info()


# In[10]:


# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)


# In[17]:


df.describe()


# ## Basic Data Cleaning & Exploration

# In[12]:


# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


# In[13]:


# Handling missing values
# Fill missing budget and revenue with median
df["budget"].fillna(df["budget"].median(), inplace=True)
df["revenue"].fillna(df["revenue"].median(), inplace=True)


# In[14]:


# Fill missing rating with mode (most frequent value)
df["rating"].fillna(df["rating"].mode()[0], inplace=True)


# In[15]:


# Drop rows where crucial text fields like title or description are missing
df.dropna(subset=["title", "description"], inplace=True)


# In[16]:


# Verify missing values after handling
print("\nMissing Values After Handling:\n", df.isnull().sum())


# In[18]:


# Distribution of numerical features
df.hist(figsize=(12, 8), bins=30)
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()


# In[22]:


# Split genres and country into lists
df["genres"] = df["genres"].astype(str).apply(lambda x: x.split(",") if x!="nan"else[])
df["country"] = df["country"].astype(str).apply(lambda x: x.split(",") if x!="nan"else[])


# In[26]:


df["genres"] = df["genres"].fillna("").astype(str).apply(lambda x:x.split(","))
df["country"] = df["country"].fillna("").astype(str).apply(lambda x:x.split(","))


# In[27]:


# Explode genres to analyze individual counts
genres_exploded = df.explode("genres")
country_exploded = df.explode("country")


# In[28]:


# Count occurrences of each genre
genre_counts = genres_exploded["genres"].value_counts().head(10)  # Top 10 genres


# In[29]:


# Count occurrences of each country
country_counts = country_exploded["country"].value_counts().head(10)  # Top 10 countries


# In[30]:


# 🔹 Top Genres Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="coolwarm")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.title("Top 10 Most Common Genres")
plt.show()


# In[31]:


# 🔹 Top Countries Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis")
plt.xlabel("Number of Movies")
plt.ylabel("Country")
plt.title("Top 10 Countries Producing Netflix Movies")
plt.show()


# ## Countrywise Performance

# In[38]:


# Group by country and calculate average rating & popularity
country_ratings = country_exploded.groupby("country")["vote_average"].mean().sort_values(ascending=False)
country_popularity = country_exploded.groupby("country")["popularity"].mean().sort_values(ascending=False)


# In[39]:


# Select top 10 countries for rating and popularity
top_countries_ratings = country_ratings.head(10)
top_countries_popularity = country_popularity.head(10)


# In[40]:


# 🔹 Highest Rated Countries
plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries_ratings.values, y=top_countries_ratings.index, palette="coolwarm")
plt.xlabel("Average Rating")
plt.ylabel("Country")
plt.title("Top 10 Highest-Rated Countries (Avg. Vote Average)")
plt.show()


# In[41]:


# Most Popular Countries
plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries_popularity.values, y=top_countries_popularity.index, palette="viridis")
plt.xlabel("Average Popularity Score")
plt.ylabel("Country")
plt.title("Top 10 Most Popular Countries (Avg. Popularity)")
plt.show()


# ## Genre Analysis

# In[32]:


# Count occurrences of each genre
genre_counts = genres_exploded["genres"].value_counts()


# In[33]:


# Display top 10 most common genres
print("Top 10 Most Common Genres:\n", genre_counts.head(10))


# In[34]:


# 🔹 Bar Chart for Visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10], palette="coolwarm")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.title("Top 10 Most Common Genres in Netflix Dataset")
plt.show()


# In[ ]:




