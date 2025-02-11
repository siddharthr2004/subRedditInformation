# 🔥 **Subreddit Market Intelligence Platform**  
### **Integrating AI, Regression Analysis & Sentiment Tracking for social media Marketing Insights**  

## **Why This Project Matters**  
🔍 Ever wondered which **products resonate the most** within different Reddit subreddits? This platform takes a **data-driven approach** to understanding subreddit trends, consumer sentiment, and product-contentment by reddit sub. By leveraging **LLM-powered analytics, regression models, and real-time tracking**, you can uncover **deep insights** into product trends, user engagement, and emerging marketing strategies.  

💡 **The Result?** An analytical system that enables **brands, businesses, and analysts** to optimize their marketing by aligning products with communities which crave their goods!.  

---  

## 🤔 **What This Platform Does**  
 **Advanced Subreddit Analysis** – Tracks, analyzes, and visualizes Reddit discussion patterns   
 **Sentiment & Emotional Analysis** – Extracts sentiment trends from millions of subreddit posts   
 **AI-Powered Insights** – Uses **LLMs + regression models** to predict **optimal marketing strategies**   
 **Automated Data Collection** – Iterative **bi-weekly updates** for continuously evolving trend tracking   
 **PostgreSQL Database Integration** – Stores **historic sentiment and demographic data**   
 **Demographic Insights** – Understand shifting user **demographics**  

---  

##  **How It Works**  
### **1️⃣ Subreddit Sentiment & Emotional Analysis**  
- Utilizes **LLMs & NLP models** to extract **emotional signals** from Reddit discussions  
- Tracks **positive, neutral, and negative sentiment trends** for brands, products, and industries  
- **Identifies high-engagement topics & common purchasing alignments** within communities  

### **2️⃣ Dynamic Regression-Based Product Matching**  
- Implements **regression analysis** to **predict the best-fit products** for each subreddit  
- Weighs sentiment fluctuations and past purchasing behaviors to **best-fit products for subs**  

### **3️⃣ Continuous Data Collection & PostgreSQL Storage**  
- Data is pulled on a **bi-weekly schedule** to showcase evolving changes  
- All analysis is stored in a **PostgreSQL database**, allowing for **historical insights & future forecasting**  

### **4️⃣ Marketing Trend Insights & Demographics**  
- Tracks **static & dynamic shifts in subreddit demographics** (age, interests, purchase intent)  
- Provides **trend evolution analysis** to highlight **emerging product opportunities**  
- Generates **visualized reports** to assist **marketers and corporations**  

---  

##  **Tech Stack**  
- **AI & NLP:** Large Language Models (LLMs), Sentiment Analysis, NLP-based classification   
- **Backend:** Python, FastAPI, Node.js  
- **Database:** PostgreSQL for structured sentiment & product analysis storage   
- **Machine Learning:** Dynamic Regression Models for trend prediction   
- **Frontend & Visualization:** React.js, D3.js, Chart.js for data visualization   
- **Data Collection:** API-based subreddit scraping, automated pipelines   

---  

##  **Key Files & Structure**  
- **`app.js`** → Backend API for sentiment & trend data retrieval  
- **`AI.py`** → Extracts information about demographics and purchasing sentiment from user comments  
- **`redditPull.js`** → Extracts commenters profiles, posts and comments, to feed into ML models  

---  

##  **Setup Guide**  
### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/yourusername/Subreddit-Analysis.git  
cd Subreddit-Analysis  
```  

### 2️⃣ Install Dependencies  
```bash  
pip install -r requirements.txt  
npm install  
```  

### 3️⃣ Set Up Database  
```bash  
psql -U your_user -d your_database -f database.sql  
```  

### 4️⃣ Start the Backend Server  
```bash  
python app.py  
```  

### 5️⃣ Start the Frontend Dashboard  
```bash  
npm start  
```  

🔗 Open in Browser: **[http://localhost:3000](http://localhost:3000)**  

---  
