import praw
import asyncpraw
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
import asyncio

#Initial example freezes the products, new implementation will defreeze and change the product embeddings by a certain amount
# ...but this will be done later

class AI:
    def __init__(self):
        # Replace these values with your Reddit app credentials
        CLIENT_ID = "IV3KzklQLKbcdr6QrNorZg"
        CLIENT_SECRET = "jv9DBrsI1GQIdQhhfUUR1B-tik2WkQ"
        USER_AGENT = "script:SiteStats:v1.0 (by /u/siddharth_reddit_acc)"
        self.reddit = asyncpraw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        # The user is passing subreddit as a command line argument, which won't work directly in Colab.
        # I will hardcode a default value for now.
        self.subreddit = "learnpython" #sys.argv[1]
        #TEST
        print(self.subreddit)

    async def getComments(self):
        authors = set()
        comments = []
        subReddit = await self.reddit.subreddit(self.subreddit)
        cleanedComments = []
        stopwordsSet = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        #Original is 3k
        async for comment in subReddit.comments(limit=30):
            if comment.author:
                authors.add(comment.author)
        #original is 400
        async for submission in subReddit.hot(limit=4):
            if submission.author:
                authors.add(submission.author)
        # Convert the set to a list before iterating asynchronously
        for author in list(authors):
            try:
                #original is 200
                async for comment in (await self.reddit.redditor(author.name)).comments.hot(limit=2):
                    comments.append(comment)
            except Exception as e:
                print(f"Error fetching comments for author {author}: {e}")
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
                if cleanedComment: # Add this check
                    cleanedComments.append(cleanedComment)
        return cleanedComments

    def getProducts(self):
        file = open("products.txt", "w")
        client = OpenAI(
        api_key="sk-proj-PtdtRgKHndHoOxGX0gqcRHinM8hEvnA-QhVw25O0vz_o84SJnYBJpMTvft" \
                "0rF9-HTyhl1vvzDT3BlbkFJPNiZV-44MS1dvalRxhnQeZZuaCq6J8nWl7rQnTT7zCiNk" \
                "259nbw7S-Dw0jQHLyuaDiZXMIlggA"
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
            file.write(ans + "\n") # Add newline for each phrase set

    def makeProductArray(self):
        arr = []
        with open('products.txt', 'r') as file:
            for line in file:
                line = re.sub(r"^\s*\d+[\.\)\-]*\s*","",line).strip() # Added strip()
                if line: # Add this check
                    arr.append(line)
        return arr

    async def getTopScores(self):
        products = np.array(self.makeProductArray())[:10]
        comments = np.array(await self.getComments())
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device = 0)
        set_seed(42)
        topScores = np.zeros(len(products))
        #original here is 200
        productBatch = np.array_split(products, 2)
        for i in range(len(comments)):
            if len(comments[i]) > 0:
                fullArray = np.array([])
                for j in range(len(productBatch)):
                  output = classifier(comments[i], productBatch[j], multi_label = True)
                  currentScores = np.array(output['scores'])
                  fullArray = np.concatenate([fullArray, currentScores])
                  time.sleep(0.001)
            topScores += fullArray 

        paired = list(zip(products, topScores))
        #original amount if 200
        topScores = sorted(paired, key = lambda x:x[1], reverse=True)[:2]
        #original amountis 200
        bottomScores = sorted(paired, key=lambda x: x[1])[:2]
        topProducts = [item[0] for item in topScores]
        bottomProducts = [item[0] for item in bottomScores]

        return {"top": topProducts, "bottom": bottomProducts}

    async def makeTensor(self):
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        comments = await self.getComments()
        if not comments: # Add check for empty comments
            print("No comments to make tensor from.")
            return None
        tensor = model.encode(comments[0], convert_to_tensor=True)
        for i in range(1, len(comments)):
            tensor = torch.add(tensor, model.encode(comments[i], convert_to_tensor=True))
        tensor = torch.div(tensor, len(comments))
        return tensor
      
    async def maximizeDotProduct(self):
        subredditTensor = await self.makeTensor()
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        topDotProductList = []
        bottomDotProductList = []
        topProducts, bottomProducts = await self.getTopScores()
        for i in range(len(topProducts)):
            #Make both product values tensors
            topProductTensor = model.encode(topProducts[i], convert_to_tensor=True)
            bottomProductTensor = model.encode(bottomProducts[i], convert_to_tensor=True)
            #Find the dot product between the subs and the tensors
            topDotProduct = torch.dot(subredditTensor, topProductTensor)
            bottomDotProduct = torch.dot(subredditTensor, bottomProductTensor)
            #
            topWeightedVal = torch.exp(-topDotProduct)
            bottomWeightedVal = torch.exp(-bottomDotProduct)
            
            topDotProductList.append((topProductTensor, topWeightedVal))
            bottomDotProductList.append((bottomProductTensor, bottomWeightedVal))
        updateDirection = torch.zeros_like(subredditTensor)
        totalWeight = 0.0
        for i in range(len(topDotProductList)):
            topProductTensor, topWeightedVal = topDotProductList[i]
            bottomProductTensor, bottomWeightedVal = bottomDotProductList[i]
            
            updateDirection += topWeightedVal * topProductTensor
            updateDirection -= bottomWeightedVal * bottomProductTensor
            
            totalWeight += topWeightedVal + bottomWeightedVal
        if totalWeight > 0:
            updateDirection = torch.div(updateDirection, totalWeight)
        return updateDirection
    
    #This will be used for finding the cosine similarity discrepencies 
    async def cosineSimilarity(self):
        subredditToAdd = await self.makeTensor() 
        subreddit = torch.nn.parameter.Parameter(data=subredditToAdd, requires_grad=True)
        maxDotProduct = await self.maximizeDotProduct()
        products = self.getProducts()
        epochs = 15
        cos = torch.nn.CosineSimilarity(1, 1e-8)
        
        for epoch in epochs:
            for product in products:
                outputProductToDot = cos(product, maxDotProduct)
                outputProductToSub = cos(product, subreddit)
                loss = (outputProductToDot - outputProductToSub) **2
                
            
           

async def main():
    test = AI()
    if test:
        print("GETTING VALS...")
        tensor = await test.maximizeDotProduct()
        print(tensor)

asyncio.run(main())






