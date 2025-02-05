import praw
import subprocess
import sys
import json
import time
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#these are the nlm scikit-learn libraries which are able to take the comments
#and place each of them within sub categories 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#This import is for the hugging-face library which allows you to take prompt
#questions to a GPT-2 model which is what I use for my sentiment pre-list
from transformers import pipeline
from datetime import datetime



class AI:
    #This essentially acts as a constructor within the python script, and it allows
    #you to convert things, from usernames to subreddits into reddit objects
    #which you can further manipulate from here
    def __init__(self):
        # Replace these values with your Reddit app credentials
        self.CLIENT_ID = "fFo133x0B-uMP6jvADZ8bg"
        self.CLIENT_SECRET = "snptPSBqbm65QIwfnB-AdkUjw-hQ1A"
        self.USER_AGENT = "script:SiteStats:v1.0 (by /u/Ancient-Opinion-4358)"
        self.reddit = praw.Reddit(
            client_id=self.CLIENT_ID,
            client_secret=self.CLIENT_SECRET,
            user_agent=self.USER_AGENT
        )

    def initialize_users(self):
        comment_array = []  # Collect all comments here

        try:
            # Get users from JS file via stdin
            input_data = sys.stdin.read()
            users = json.loads(input_data)  # Load usernames from JSON

            for username in users:
                try:
                    # Convert username to Redditor object
                    user = self.reddit.redditor(username)

                    # Fetch comments from user
                    for comment in user.comments.new(limit=300):
                        comment_array.append(comment.body)  # Append comment body
                        print(f"User: {username}, comment processed")
                    
                    # Pause for 1 second between users
                    time.sleep(1)

                except Exception as user_error:
                    print(f"Error processing user {username}: {user_error}")

            return comment_array

        except Exception as e:
            print(f"Error: {e}")
            return []
        
    
    def cleanComments(self):
        #First we'll add some defintiions within our nltk system
        stopwords = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        dirtyComments = self.initialize_users()
        cleanedComments = [];
        
        for comments in dirtyComments:
            #Check if the comment is a string the syntax below is how to check this
            if isinstance(comments, str):
                #Remove URLs
                comments = re.sub(r"http\S+|www\S+|https\S+", "", comments)
                #Remove non-alphabetic characters and numbers, and convert to lowercase
                comments = re.sub(r"[^a-z\s]", "", comments.lower())
                #Remove extra spaces
                comments = re.sub(r"\s+", " ", comments).strip()
                #Tokenize the cleaned comment
                tokens = word_tokenize(comments)
                #Remove stop words
                filtered_tokens = [word for word in tokens if word not in stopwords]
                #Apply lematization (Running -> Run)
                lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
                #Recombine the tokens into a cleaned string (optional)
                cleaned_comment = " ".join(lemmatized)
                #Append the cleaned comment to the list
                cleanedComments.append(cleaned_comment);

        # Group all comments which are similar into categories and name them
    def group_comments(comments):
        # Step 1: Convert comments into a matrix of word counts (Vectorization)
        # This vectorizes the comments and creates a matrix where rows = comments and columns = unique words
        vectorizer = CountVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(comments)

        '''
        How topic sampling works: First, each comment is placed into a vector along with unique words. 
        Then, each of the comments is mapped out to the different words which exist. 
        Similarities in the mapping are identified, and these similarities are grouped into topics, numbered 0 to n.
        Words are assigned weight based on both their relevance within overall uniqueness and importance within the comment itself.
        This is then used to derive an analysis of similar comments and topic clustering.
        '''

        # Step 2: Apply Latent Dirichlet Allocation (LDA) to identify topics
        # Specify the number of topics (n_components=5 for simplicity) and set random_state for reproducibility
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(matrix)  # Fit the LDA model to the word-count matrix

        # Step 3: Assign each comment to a topic
        # Transform the word-count matrix into topic probabilities for each comment
        topic_probabilities = lda_model.transform(matrix)
        comment_topics = []

        for probabilities in topic_probabilities:
            # Find the topic with the highest probability for this comment
            top_topic = probabilities.argmax()  # This identifies the most relevant topic index
            '''
            At this stage, each comment is assigned a topic within the vector table. The topic index is stored 
            in a separate array. For example, the comment "I love school" might be matched to topic number 4.
            '''
            comment_topics.append(top_topic)

        # Step 4: Extract representative words for each topic to give them descriptive names
        feature_names = vectorizer.get_feature_names_out()  # Get the list of unigrams from the vectorization process
        topic_names = []

        for topic_idx in range(len(lda_model.components_)):
            # Each topic contains word weights
            topic = lda_model.components_[topic_idx]
            # Sort the words by their weight (importance) in ascending order
            sorted_indices = topic.argsort()
            # Get the top 5 words by weight (most important for the topic)
            top_indices = sorted_indices[-5:]  # -5: selects the last 5 items (highest weights)
            top_words = []

            # Reverse the order to list the most important words first
            for index in reversed(top_indices):
                top_words.append(feature_names[index])  # Map indices to the corresponding words

            # Combine the top words into a single string to create the topic's name
            topic_names.append(" ".join(top_words))

        # Step 5: Group comments based on their assigned topics
        # Create a dictionary for topic-based grouping
        grouped_comments = {
            name: {"listwords": [], 
                    "comments": [], 
                    "sentiment": [],
                    "emotion": []
                    } 
                    for name in topic_names
            }  

        for i, topic_id in enumerate(comment_topics):
            # Get the name of the topic for the assigned topic ID
            topic_name = topic_names[topic_id]  
            # Add the comment to the appropriate topic group
            grouped_comments[topic_name]["comments"].append(comments[i]) 

        return grouped_comments

    '''
    After parsing out the groups - within the 5 words extracted from the 500
    or so groups, find a further 500 phrases, products and words which are 
    commonly associted within the words which exist inside the group. Here we can
    use the "hugging-face" transformer which takes in any model (like GPT), and
    allows you to input a text which it will then output information for said text
    '''
    
    def getKeyWords(self): 
        # Firstly get all groups which exist
        groupedComments = self.group_comments()  # Ensure this returns the correct structure

        # Initialize the text generation pipeline outside the loop
        generator = pipeline("text-generation", model="gpt2")

        # Iterate through the grouped comments
        for topic_name, group_data in groupedComments.items():
            # Extract the key words for the topic (assuming they are stored in 'listWords')
            top_words = ", ".join(group_data["listWords"])  # Convert list to a comma-separated string

            # Create the prompt
            prompt = f"""
                Generate a list of 400 common words, phrases, and tangible 
                products that are strongly associated with the following 
                terms: {top_words}. Focus specifically on sellable, physical 
                products wherever possible. Be as specific and detailed as you can 
                in generating this list.
            """

            # Generate the text
            response = generator(prompt, max_length=200, num_return_sequences=1)

            # Extract the generated text and append it to 'listWords'
            generated_text = response[0]["generated_text"]  # Access the generated text
            group_data["listWords"].append(generated_text)  # Append to 'listWords'

        return groupedComments

    '''
    This next portion will focus on the senitment analysis of each of the groups
    and how they pertain to the word stream which was given out previously. For this
    we will use the BERT model
    '''
    def sentimentAndEmotionAnalysis(self):
        groupedComments = self.getKeyWords()
        #initialize the sentiment pipelines
        sentimentAnalyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        emotionAnalyzer = pipeline(
            "text-classification",
            model ="j-harmann/emotion-english-distilroberta-base"
        )
        # Iterate through the groups
        for group_name, group_data in groupedComments.items():
            comments = " ".join(group_data["comments"])  # Aggregate all comments into a single text block
            listwords = group_data["listwords"]
            for word in listwords:
                # Create a context block for the word
                context = f"The following block of text is about {word}: {comments}"
                # Perform sentiment analysis
                sentiment = sentimentAnalyzer(context)
                # Perform emotional analysis
                emotion = emotionAnalyzer(context)
                # Append sentiment results
                group_data["sentiment"].append({
                    "listword": word,
                    "sentiment": sentiment[0]["label"],  # Positive/Negative/Neutral
                    "confidence": sentiment[0]["score"]  # Confidence score
                })
                # Append emotion results
                group_data["emotion"].append({
                    "listword": word,
                    "emotion": emotion[0]["label"],  # Emotion label
                    "confidence": emotion[0]["score"]  # Confidence score
                })
        return groupedComments
    
    def scoreAggregate(self):
        groupedComments = self.sentimentAndEmotionAnalysis()
        returnedScores = {}
        #Finalized library:
        finalScores = {}
        # Iterate through groups
        for group_name, group_data in groupedComments.items():
            # Initialize lists for positive and negative scores for this group

            finalScores[group_name] = {
                "positiveSentiment": [],
                "positiveEmotion": [],
            }
    
            for word_data in group_data["listwords"]:  # Iterate through listwords
                # Check sentiment and append to appropriate list
                if word_data["emotion"] == "POSITIVE":
                    finalScores[group_name]["positiveEmotion"].append(word_data)
                elif word_data["sentiment"] == "POSITIVE":
                    finalScores[group_name]["positiveSentiment"].append(word_data)
                

                #Now sort the values within commentPosScores and commentNegScores for each
                # Sort each list in descending order of confidence scores
                finalScores[group_name]["positiveEmotion"] = sorted(
                    finalScores[group_name]["positiveEmotion"], key=lambda x: x["confidence"], reverse=True
                )
                finalScores[group_name]["positiveSentiment"] = sorted(
                    finalScores[group_name]["positiveSentiment"], key=lambda x: x["confidence"], reverse=True
                )
        
        for group_name, group_data in finalScores.items:
            
            returnedScores[group_name] = {
                "groupName": group_name,  # Store the group's name
                "topFiveEmotion": [],  # Initialize an empty list for the top 5 scores
                "topFiveSentiment": []
            }
            # Add the top 5 positive emotions (already sorted)
            if "positive_emotions" in group_data:
                returnedScores[group_name]["topFiveEmotion"] = group_data["positive_emotions"][:5]

            # Add the top 5 positive sentiments (already sorted)
            if "positive_sentiments" in group_data:
                returnedScores[group_name]["topFiveSentiment"] = group_data["positive_sentiments"][:5]

        return returnedScores
    
    '''
        We have now completed the sentiment/emotional analysis on the values provided
        to the subreddit. At this step - we will not perform a demographic analysis
        which we will then feed into the regression model to find best fit products for 
        the subreddit
    '''

    def extractAge(self):
        comments = self.cleanComments()  # Assuming this returns a list of comments

        AgeToGeneration = {
         "genZ": (11, 26),
            "millenial": (27, 40),
            "genX": (41, 57),
            "boomer": (58, 100)
        }

        agePatterns = [
            r"\bI[']?m (\d{1,2}) years? old\b",
            r"\bI[']?m (\d{1,2})\b",
            r"\bas a (\d{1,2}) year? old?\b",
            r"\bturned (\d{1,2}) years?\b",
            r"\babout to turn (\d{1,2})\b",
            r"\bturning (\d{1,2})\b"
        ]

        yearPatterns = [
            r"\bborn in (\d{4})",
        ]

        generationPatterns = [
            r"\bas a (millennial|gen z|gen x|boomer)\b"
        ]

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidateLabels = ["<26", "27-40", "41-57", "58-130"]

        extractedInformation = {
            "genZ": [],
            "millenial": [],
            "genX": [],
            "boomer": []
        }

        # Match comments to age based on regex
        for comment in comments:
            for pattern in agePatterns:
                match = re.search(pattern, comment)
                if match:
                    age = int(match.group(1))
                    for generation, (lower, upper) in AgeToGeneration.items():
                        if lower <= age <= upper:
                            extractedInformation[generation].append(age)

            for pattern in yearPatterns:
                match = re.search(pattern, comment)
                if match:
                    birth_year = int(match.group(1))
                    age = datetime.now().year - birth_year
                    for generation, (lower, upper) in AgeToGeneration.items():
                        if lower <= age <= upper:
                            extractedInformation[generation].append(age)

            for pattern in generationPatterns:
                match = re.search(pattern, comment, re.IGNORECASE)
                if match:
                    generation_label = match.group(1).lower()
                    if generation_label == "gen z":
                        extractedInformation["genZ"].append(generation_label)
                    elif generation_label == "millennial":
                        extractedInformation["millenial"].append(generation_label)
                    elif generation_label == "gen x":
                        extractedInformation["genX"].append(generation_label)
                    elif generation_label == "boomer":
                        extractedInformation["boomer"].append(generation_label)

        # AI Predicted Age
        AiAges = {
            "genZ": 0.0, 
            "millenial": 0.0,
            "genX": 0.0,
            "boomer": 0.0
        }

        # Process AI predictions for all comments
        for comment in comments:
            AiResults = classifier(comment, candidateLabels)  # Classify each comment

            for label, confidence in zip(AiResults["labels"], AiResults["scores"]):
                # Extract first number from label
                age = int(label.split("-")[0].replace("<", "").replace(">", ""))

                # Aggregate confidence into respective generation
                if age < 26:
                    AiAges["genZ"] += confidence
                elif 27 <= age <= 40:
                    AiAges["millenial"] += confidence
                elif 41 <= age <= 57:
                    AiAges["genX"] += confidence
                elif age >= 58:
                    AiAges["boomer"] += confidence

        # Normalize AI confidence scores (convert to scale of 0-1)
        normalized_AiAges = {key: value / 100 for key, value in AiAges.items()}

        # Compute final scores
        genZ = (0.5 * len(extractedInformation["genZ"])) + (0.5 * normalized_AiAges["genZ"])
        millenial = (0.5 * len(extractedInformation["millenial"])) + (0.5 * normalized_AiAges["millenial"])
        genX = (0.5 * len(extractedInformation["genX"])) + (0.5 * normalized_AiAges["genX"])
        boomer = (0.5 * len(extractedInformation["boomer"])) + (0.5 * normalized_AiAges["boomer"])

        return {
            "genZ": genZ,
            "millenial": millenial,
            "genX": genX,
            "boomer": boomer
        }

        #now we will extract all of the locations
    def extractLocations(self):
        comments = self.cleanComments()

        classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
        # Define candidate locations
        stateCandidateLabels = [
            # U.S. States (20 most common ones)
            "California", "Texas", "New York", "Florida", "Illinois", "Pennsylvania", "Ohio",
            "Georgia", "North Carolina", "Michigan", "New Jersey", "Virginia", "Washington",
            "Arizona", "Massachusetts", "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin",
        ]

            # Countries (30 diverse countries)
        countryCandidateLabels = [
            "USA", "Canada", "Mexico", "United Kingdom", "France", "Germany", "Italy", "Spain",
            "Australia", "India", "China", "Japan", "South Korea", "Brazil", "Argentina", "Belgium",
            "Russia", "Turkey", "Egypt", "Saudi Arabia", "Nigeria", "Kenya", "Sweden", "Norway", "Netherlands",
            "Switzerland", "Poland", "Ukraine", "Indonesia", "Philippines"
        ]

        continentCandidateLabels = [
            "North America", "South America", "Europe", "Asia", "Africa", "Australia", "Antarctica"
        ]

        AIResultsState = classifier(comments, stateCandidateLabels);
        AIResultsCountry = classifier(comments, countryCandidateLabels)
        AIResultsContinent = classifier(comments, continentCandidateLabels)

        statConfidence = {}, countryConfidence = {}, continentConfidence = {}

        









            









        



        

    





        




    
                

    
     