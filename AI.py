import praw
import subprocess
import sys
import json
import time
import re
import openai
import os
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
    
    def getVals(self):
        authors = set()
        comments = []
        subReddit = self.reddit.subreddit(self.subreddit)
        cleanedComments = []
        stopwordsSet = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        vectorizer = CountVectorizer()
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
        matrix = vectorizer.fit_transform(cleanedComments)


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
        stopwordsSet = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        dirtyComments = self.initialize_users()
        cleanedComments = []
        
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
                filtered_tokens = [word for word in tokens if word not in stopwordsSet]
                #Apply lematization (Running -> Run)
                lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
                #Recombine the tokens into a cleaned string (optional)
                cleaned_comment = " ".join(lemmatized)
                #Append the cleaned comment to the list
                cleanedComments.append(cleaned_comment);

        # Group all comments which are similar into categories and name them
    def group_comments(self):
        # Step 1: Convert comments into a matrix of word counts (Vectorization)
        # This vectorizes the comments and creates a matrix where rows = comments and columns = unique words
        comments = self.cleanComments()
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
            cleaned_text = generated_text.split("\n")[:500]  # ✅ Only keeps the first 500 words
            group_data["listwords"].extend(cleaned_text)

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
    
            for word_data in group_data["listwords"]:
                if word_data in group_data["emotion"] and group_data["emotion"][word_data]["emotion"] == "POSITIVE":
                    finalScores[group_name]["positiveEmotion"].append(group_data["emotion"][word_data])

                if word_data in group_data["sentiment"] and group_data["sentiment"][word_data]["sentiment"] == "POSITIVE":
                    finalScores[group_name]["positiveSentiment"].append(group_data["sentiment"][word_data])

                

                #Now sort the values within commentPosScores and commentNegScores for each
                # Sort each list in descending order of confidence scores
                finalScores[group_name]["positiveEmotion"] = sorted(
                    finalScores[group_name]["positiveEmotion"], key=lambda x: x["confidence"], reverse=True
                )
                finalScores[group_name]["positiveSentiment"] = sorted(
                    finalScores[group_name]["positiveSentiment"], key=lambda x: x["confidence"], reverse=True
                )
        
        for group_name, group_data in finalScores.items():
            
            returnedScores[group_name] = {
                "groupName": group_name,  # Store the group's name
                "topFiveEmotion": [],  # Initialize an empty list for the top 5 scores
                "topFiveSentiment": []
            }
            ######THIS HAS BEEN CHANGED TO 15 EMOTIONS AND SENTIMENTS INSTEAD OF 5#########
            # Add the top 5 positive emotions (already sorted)
            if "positive_emotions" in group_data:
                returnedScores[group_name]["topFiveEmotion"] = group_data["positiveEmotion"][:15]

            # Add the top 5 positive sentiments (already sorted)
            if "positive_sentiments" in group_data:
                returnedScores[group_name]["topFiveSentiment"] = group_data["positiveSentiment"][:15]

        return returnedScores
    
    ###################################################################################################################
                        ###################NEW_SECTION_DEMOGRAPHICS#########################
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


    def extractLocations(self):
        comments = self.cleanComments()

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Define candidate locations
        stateCandidateLabels = [
            "California", "Texas", "New York", "Florida", "Illinois", "Pennsylvania", "Ohio",
            "Georgia", "North Carolina", "Michigan", "New Jersey", "Virginia", "Washington",
            "Arizona", "Massachusetts", "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin"
     ]

        countryCandidateLabels = [
          "USA", "Canada", "Mexico", "United Kingdom", "France", "Germany", "Italy", "Spain",
           "Australia", "India", "China", "Japan", "South Korea", "Brazil", "Argentina", "Belgium",
           "Russia", "Turkey", "Egypt", "Saudi Arabia", "Nigeria", "Kenya", "Sweden", "Norway", "Netherlands",
          "Switzerland", "Poland", "Ukraine", "Indonesia", "Philippines"
      ]

        continentCandidateLabels = [
          "North America", "South America", "Europe", "Asia", "Africa", "Australia", "Antarctica"
        ]

        # ✅ Corrected `hypothesis_template` usage
        AIResultsState = classifier(comments, stateCandidateLabels, hypothesis_template="This person is from {}.")
        AIResultsCountry = classifier(comments, countryCandidateLabels, hypothesis_template="This person is from {}.")
        AIResultsContinent = classifier(comments, continentCandidateLabels, hypothesis_template="This person is from {}.")

        # Initialize count dictionaries
        stateCounts, countryCounts, continentCounts = {}, {}, {}

        # Loop through comments to store counts
        for i, comment in enumerate(comments):
            if AIResultsState[i]["scores"][0] > 0.8:
                location = AIResultsState[i]['labels'][0]
                stateCounts[location] = stateCounts.get(location, 0) + 1
            if AIResultsCountry[i]["scores"][0] > 0.8:
                location = AIResultsCountry[i]['labels'][0]
                countryCounts[location] = countryCounts.get(location, 0) + 1
            if AIResultsContinent[i]["scores"][0] > 0.8:
                location = AIResultsContinent[i]['labels'][0]
                continentCounts[location] = continentCounts.get(location, 0) + 1

        # 🔹 Sort dictionaries and return the top 5
        sortedStateCounts = sorted(stateCounts.items(), key=lambda item: item[1], reverse=True)[:5]  
        sortedCountryCounts = sorted(countryCounts.items(), key=lambda item: item[1], reverse=True)[:5]  
        sortedContinentCounts = sorted(continentCounts.items(), key=lambda item: item[1], reverse=True)[:5]  

        return sortedStateCounts, sortedCountryCounts, sortedContinentCounts
    
    from transformers import pipeline

    from transformers import pipeline

    def extractSocioEconomicDemographics(self):
        comments = self.cleanComments()

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # **Step 1: Broad Categories**
        employmentLabels = ["Employed", "Unemployed"]
        educationLabels = ["College Graduate", "No College"]

        # **Step 2: Detailed Categories**
        occupationLabels = [
            # White-collar
            "Software Engineer", "Data Scientist", "Financial Analyst", "Marketing Manager",
            "Sales Representative", "Consultant", "Accountant", "Project Manager",
            "Lawyer", "Doctor", "Nurse", "Professor", "Architect", "Journalist",

            # Blue-collar
            "Factory Worker", "Construction Worker", "Electrician", "Plumber", "Mechanic",
            "Truck Driver", "Welder", "Carpenter", "HVAC Technician", "Auto Technician",

            # Service jobs
            "Retail Worker", "Customer Service Representative", "Barista", "Waiter", "Bartender",
            "Chef", "Janitor", "Hotel Staff", "Cashier", "Delivery Driver",

            # Creative & independent work
            "Artist", "Writer", "Musician", "YouTuber", "Influencer", "Game Developer",
            "Photographer", "Graphic Designer", "Filmmaker", "Twitch Streamer",

            # Military, government, and law enforcement
            "Military Personnel", "Police Officer", "Firefighter", "Government Employee",
            "Politician", "Diplomat", "CIA Agent", "FBI Agent",

            # Self-employed
            "Self-Employed", "Entrepreneur", "Freelancer", "Gig Worker",
        ]

        incomeLabels = ["Low Income", "Lower Middle Class", "Middle Class", "Upper Middle Class", "High Income"]

        # **Step 1: Classify Broad Categories**
        AIResultsEmployment = classifier(comments, employmentLabels, hypothesis_template="This person is {}.")
        AIResultsEducation = classifier(comments, educationLabels, hypothesis_template="This person has {} education.")
        AIResultsIncome = classifier(comments, incomeLabels, hypothesis_template="This person belongs to the {} income group.")

        # Dictionaries to store counts
        employmentCounts, educationCounts, occupationCounts, incomeCounts = {}, {}, {}, {}

        employedCount = 0
        unemployedCount = 0

        for i, comment in enumerate(comments):
            # **Step 1: Store broad category classifications**
            if AIResultsEmployment[i]["scores"][0] > 0.8:
                category = AIResultsEmployment[i]['labels'][0]
                employmentCounts[category] = employmentCounts.get(category, 0) + 1

                # Track total counts
                if category == "Employed":
                    employedCount += 1
                else:
                    unemployedCount += 1

            if AIResultsEducation[i]["scores"][0] > 0.8:
                category = AIResultsEducation[i]['labels'][0]
                educationCounts[category] = educationCounts.get(category, 0) + 1

            # **Step 2: Classify Occupation Only for "Employed"**
            if category == "Employed":
                AIResultsOccupation = classifier(comment, occupationLabels, hypothesis_template="This person works as a {}.")
                if AIResultsOccupation["scores"][0] > 0.8:
                    job = AIResultsOccupation['labels'][0]
                    occupationCounts[job] = occupationCounts.get(job, 0) + 1

            # **Step 2: Classify Income for Everyone**
            AIResultsIncomeForComment = classifier(comment, incomeLabels, hypothesis_template="This person belongs to the {} income group.")
            if AIResultsIncomeForComment["scores"][0] > 0.8:
                incomeCategory = AIResultsIncomeForComment['labels'][0]
                incomeCounts[incomeCategory] = incomeCounts.get(incomeCategory, 0) + 1

        # **Sort and Return the Top 5 for Each**
        sortedEmploymentCounts = sorted(employmentCounts.items(), key=lambda item: item[1], reverse=True)
        sortedEducationCounts = sorted(educationCounts.items(), key=lambda item: item[1], reverse=True)[:5]
        sortedOccupationCounts = sorted(occupationCounts.items(), key=lambda item: item[1], reverse=True)[:5]
        sortedIncomeCounts = sorted(incomeCounts.items(), key=lambda item: item[1], reverse=True)[:5]

        return {
            "Employment Breakdown": sortedEmploymentCounts,
            "Total Employed": employedCount,
            "Total Unemployed": unemployedCount,
            "Top Education Levels": sortedEducationCounts,
            "Top Occupations": sortedOccupationCounts,
            "Top Income Brackets": sortedIncomeCounts
        }
    
    #extract gender, political affiliation, race, lgbtq+, religion grouping
    def extractIdentity(self):
        comments = self.cleanComments()

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Different grouping
        categories = {
            "race": ["white", "black", "asian", "latino", "hispanic", "indian", "arab"],
            "gender": ["man", "woman", "nonbinary", "trans", "female", "male"],
            "lgbt": ["gay", "lesbian", "bisexual", "pansexual", "transgender", "queer"],
            "politics": ["liberal", "conservative", "leftist", "right-wing", "democrat", "republican"],
            "religion": ["Christian", "Muslim", "Jewish", "Hindu", "Buddhist", "Atheist"]
        }

        # Initialize dictionary to hold scores
        identityScores = {category: {label: 0 for label in labels} for category, labels in categories.items()}

        for comment in comments:
           # Category is race, gender, etc.; labels is dictionary of {"white": 0, "black": 0, ...}
           for category, labels in identityScores.items():
               # Classify using all labels in the category
               result = classifier(comment, list(labels.keys()), multi_label=True)

               # Loop through classification results
               for label, score in zip(result["labels"], result["scores"]): 
                   if score > 0.5:
                       identityScores[category][label] += 1  # Fix dictionary update

        # Sort and get top 2 labels per category
        topLabelsPerCategory = {}
        for category, labels in identityScores.items():
            sortedLabels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # Fix sorting
            topLabelsPerCategory[category] = sortedLabels[:2]  # Extract top 2

        return topLabelsPerCategory  # Move return outside the loop

    def prepareDataForRegression(self):
        # Extract all demographic data
        ageData = self.extractAge()
        locationData = self.extractLocations()
        socioEconomicData = self.extractSocioEconomicDemographics()
        identityData = self.extractIdentity()
        sentimentAndEmotionData = self.sentimentAndEmotionAnalysis()  

        promptFull = f"""
        Based on the following information which has been extracted from these subreddits
        your goal and job is to product 15 best fit marketing products which would do the 
        best on this subreddit, and would garner the most positive reactions and "sellability"
        on this sub. 

        I will first input the 15 most positive reactions  to different common marketing and/or 
        business terms which the subreddit had. {sentimentAndEmotionData}

        I will now input a bunch of demographic data which I had collected from this subreddit:
        age: {ageData}, 
        location: {locationData},
        socioEconomic information: {socioEconomicData},
        Identity information (race, sex, gender, sexuality, religion): {identityData}
        """
        
        openai.api_key = (
            "sk-proj-hKkIFM7ijRoljf66Xj8wVbzfoxLdQ0L8vqlkFrG340nj34qwA4JfNgZV7"
            "ZVp1OA8AC0WkwU46uT3BlbkFJwmEeLKY2fOgErzLXQ9itE4NsmlvOEMs6mN5ul6b_j"
            "vsEQ_iiWi8YGyi6M9Qd8rAAAdlPsLH9MA"
        )

        
        response = openai.completions.create(  # ✅ Correct function
            model="gpt-3.5-turbo-instruct",
            prompt=promptFull,  # ✅ Correct syntax
            max_tokens=100  # Optional limit on response length
        )

        return response






        


    




            









        



        

    





        




    
                

    
     