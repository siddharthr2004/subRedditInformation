## **Subreddit-Product matching tool** 
Using AI, sentiment analysis and neural networks to match subreddits to their best fit products

## **Features:**
- Uses gradient testing from thousands of comments to match best fit products
- Synthesizes and outputs large list ranked with best products to be marketed on each sub 
- Holds storage information on best product within each sub, and updates weekly

##  **Key Files & Structure**  
- **`app.js`**  Backend API used for web rendering (in progress) 
- **`AI.py`**   Extracts the best fit products for the subreddit while storing which products best match the sub

---  

##  **Setup Guide**  
### Clone the Repository  
```bash  
git clone https://github.com/siddharthr2004/Subreddit-Analysis.git  
cd Subreddit-Analysis  
```  

###  **Install Dependencies  
```bash  
pip install -r requirements.txt  
npm install  
```  

### 4️⃣ Start the Backend Server  
```bash  
python AI.py  
```  
