import praw
import subprocess
import sys
import json
import time
import re
import openai
import os
import nltk
import numpy as np
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from transformers import set_seed
from datetime import datetime



class AI:
    def __init__(self):
        # Replace these values with your Reddit app credentials
        CLIENT_ID = "fFo133x0B-uMP6jvADZ8bg"
        CLIENT_SECRET = "snptPSBqbm65QIwfnB-AdkUjw-hQ1A"
        USER_AGENT = "script:SiteStats:v1.0 (by /u/Ancient-Opinion-4358)"
        self.reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        self.subreddit = sys.argv[1]
    
    def getComments(self):
        authors = set()
        comments = []
        subReddit = self.reddit.subreddit(self.subreddit)
        cleanedComments = []
        stopwordsSet = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        for comment in subReddit.comments(limit=3000):
            if comment.author:
                authors.append(comment.author)
        for submission in subReddit.hot(limit=400):
            if submission.author:
                authors.append(submission.author)
        for author in authors:
            for comment in author.comments.hot(limit=200):
                comments.append(comment)
        for comment in comments:
            text = comment.body
            if(isinstance(text, str)):
                cleanedComment = re.sub(r"http\S+|www\S+|https\S+", "", text)
                cleanedComment = re.sub(r"^[a-z]\s", "", cleanedComment.lower())
                cleanedComment = re.sub(r"\s+", "", cleanedComment.strip())
                tokens = word_tokenize(cleanedComment)
                commentsNew = [words for words in tokens if words not in stopwordsSet]
                lemmatizedWord = [lemmatizer.lemmatize(words) for words in commentsNew]
                cleanedComment = " ".join(lemmatizedWord)
                cleanedComments.append(cleanedComment)
                return cleanedComments
        
    def getProducts(self):
        generator = pipeline("text-generation", model="gpt2-large")
        set_seed(42)
        vector = generator('''"Generate 2000 distinct short phrases that reflect individual lifestyle preferences and consumer product usage. 
                           Each phrase should relate to personal choices, routines, or experiences involving specific product categories such as 
                           fitness, electronics, fashion, nutrition, home decor, travel gear, or digital services. Phrases should resemble 
                           real-world expressions, reviews, or habits shared by consumers online.''', max_length = 30, num_return_sequences=2000, 
                           temperature=0.8, top_p=0.9, top_k=50)
        phrases = [phrase['generated_text'] for phrase in vector]
        return phrases
    
    def getTopScores(self):
        products = self.getProducts()
        comments = self.getComments()
        