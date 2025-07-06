import praw
import subprocess
import sys
import json
import time
import re
import openai
import os
import nltk
import torch
import math
import numpy as np
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from transformers import set_seed
from sentence_transformers import SentenceTransformer
from datetime import datetime

#Initial example freezes the products, new implementation will defreeze and change the product embeddings by a certain amount
# ...but this will be done later

class AI:
    def __init__(self):
        # Replace these values with your Reddit app credentials
        CLIENT_ID = "IV3KzklQLKbcdr6QrNorZg"
        CLIENT_SECRET = "jv9DBrsI1GQIdQhhfUUR1B-tik2WkQ"
        USER_AGENT = "script:SiteStats:v1.0 (by /u/siddharth_reddit_acc)"
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
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        set_seed(42)
        topScores = np.zeros(len(products))
        for comment in comments:
            output = classifier(comment, products, multi_label=True)
            currentScores = np.array(output['scores'])
            topScores += currentScores
        paired = list(zip(products, topScores))
        #TESTING ONLY
        print([vals for vals, product in paired])
        topScores = sorted(paired, key = lambda x:x[1], reverse=True)[200:]
        bottomScores = sorted(paired, key = lambda x:x[1], reverse=False)[:200]
        topProducts = [item[0] for item in topScores]
        bottomProducts = [item[0] for item in bottomScores]
        return {"top": topProducts, "bottom": bottomProducts}
    
    #Use weighted pooling going into the future
    def makeTensor(self):
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        comments = self.getComments()
        tensor = model.encode(comments[0], convert_to_tensor=True) 
        for i in range(1, len(comments)):
            tensor = torch.add(tensor, model.encode(comments[i], convert_to_tensor=True))
        tensor = torch.div(tensor, len(comments))
        return tensor

    def maximizeDotProduct(self):
        subredditTensor = self.makeTensor()
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        dotProductList = []
        topProducts, bottomProducts = self.getTopScores()
        for product in topProducts:
            productTensor = model.encode(product, convert_to_tensor=True)
            dotProduct = torch.dot(subredditTensor, productTensor)
            weightedVal = torch.exp(-dotProduct)
            dotProductList.append((productTensor, weightedVal))
        updateDirection = torch.zeros_like(subredditTensor)
        totalWeight = 0.0
        for productTensor, weightedVal in dotProductList:
            updateVal = torch.mul(productTensor, weightedVal)
            updateDirection.append(updateVal)     
            totalWeight += weightedVal
        if totalWeight > 0:
            updateDirection = torch.div(updateDirection, totalWeight)
    
test = AI()
test.getTopScores()
            



    
    