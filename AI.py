import praw
import subprocess
import sys
import json
import time
import re
from openai import OpenAI
import os
import nltk
import torch
import math
import numpy as np
nltk.download("punkt", quiet=True)
nltk.download('stopwords', quiet = True)
nltk.download('punkt_tab', quiet = True)
nltk.download('wordnet', quiet = True)
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
        #TEST
        print(self.subreddit) 
    
    def getComments(self):
        print("came here 1")
        authors = set()
        comments = []
        subReddit = self.reddit.subreddit(self.subreddit)
        cleanedComments = []
        stopwordsSet = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        #Original is 3k
        for comment in subReddit.comments(limit=30):
            if comment.author:
                authors.add(comment.author)
        #original is 400
        for submission in subReddit.hot(limit=4):
            if submission.author:
                authors.add(submission.author)
        for author in authors:
            try:
                #original is 200
                for comment in (author.comments.hot(limit=2)):
                    comments.append(comment)
            except Exception as e:
                print(f"Error fetching comments for author {author}: {e}")
        print("dont adding values to comment")
        for comment in comments:
            text = comment.body
            if(isinstance(text, str)):
                cleanedComment = re.sub(r"http\S+|www\S+|https\S+", "", text)
                cleanedComment = cleanedComment.strip()
                cleanedComment = re.sub(r"\s+", " ", cleanedComment)
                tokens = word_tokenize(cleanedComment)
                commentsNew = [words for words in tokens if words not in stopwordsSet]
                lemmatizedWord = [lemmatizer.lemmatize(words) for words in commentsNew]
                cleanedComment = " ".join(lemmatizedWord)
                cleanedComments.append(cleanedComment)
        return cleanedComments
    
    def getProducts(self):
        file = open("products.txt", "w")
        client = OpenAI(
        api_key="sk-proj-PtdtRgKHndHoOxGX0gqcRHinM8hEvnA-QhVw25O0vz2_o84SJnYBJpMTvft" \
                "0rF9-HTyhl1vvzDT3BlbkFJPNiZV-44MS1dvalRxhnQeZZuaCq6J8nWl7rQnTT7zCiNk" \
                "259nbw7S-Dw0jQHLyuaDiZXMIvlgA"
        )
        for _ in range(8):
            ans = []
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "user",
                        "content": "Generate 100 short descriptive identity phrases. Each phrase should "
                        "combine traits, labels, or roles that describe types of people. These can include"
                        "political, emotional, social, racial, occupational, or behavioral dimensions."

                        "Examples:"
                        "- liberal activist"
                        "- conservative parent"
                        "- black entrepreneur"
                        "- white suburban voter"
                        "- blue-collar mechanic"
                        "- logical analyst"
                        "- emotion-driven leader"
                        "- socially awkward genius"
                        "- hyper-organized scheduler"
                        "- risk-taking gambler"
                    }
                ],
            )
            ans = response.choices[0].message.content	
            file.write(ans)

    def makeProductArray(self):
        arr = []
        with open('products.txt', 'r') as file:
            for line in file:
                arr.append(re.sub(r"^\s*\d+[\.\)\-]*\s*","",line))
        return arr
    
    def getTopScores(self):
        products = np.array(self.makeProductArray())
        comments = np.array(self.getComments())
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device = 1)
        set_seed(42)
        topScores = np.zeros(len(products))
        productBatch = np.array_split(products, 200)
        for i in range(len(comments)):
            fullArray = np.array([])
            for j in range(len(productBatch)):
                print(len(productBatch))
                print("trying to print vals from productBatch")
                print("printing the comment", comments[i])
                for product in productBatch[j]:
                    print(product)
                output = classifier(comments[i], productBatch[j], multi_label = True)
                currentScores = np.array(output['scores'])
                #TEST
                print(currentScores)
                fullArray = np.concatenate(fullArray, currentScores)
                time.sleep(0.1)
            topScores += fullArray
        paired = list(zip(products, topScores))
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
if (test):
    print("GETTING VALS...")
    arr = test.getTopScores()
    
            



    
    