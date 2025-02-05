//THIS CLASS FIRST PULLS ALL OF THE COMMENTS FOR ALL 500 USERS WHICH GOT FROM THE redditPull.js CLASS. IT WILL THEN INCREMENTALLY ADD
//THIS INFORMATION INTO AN AI MACHINE WHERE AN IDEA OF THE BEST PRODUCTS TO ADD WILL BE GARNDERED
const https = require('https');
const RedditPull = require('./redditPull.js');

class userPull {
    // Assuming fetchRedditToken() and orderData() are implemented correctly
    constructor (subreddit) {
        this.subreddit = subreddit;
    }


    async getUserComments() {
        const redditPull = new RedditPull(this.subreddit);
        const token = await redditPull.fetchRedditToken();
        //FOR TESTING PURPOSES THIS HAS BEEN COMMENTED OUT
        //const data = await redditPull.orderData();
        //FOR TESTING PURPOSES WE WILL ADD HARDCODED USERS IN
        const data = [];
        data.push('ReginaldDoom');
        data.push('Loose-Ad7862');
        data.push('lana_del_rey_lover69');
        
        const userComments = [];

        // Create a promise for each user
        const promises = data.map((user) => {
            const options = {
                hostname: 'oauth.reddit.com',
                path: `/user/${user}/comments/top.json?limit=3&t=all`,
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'User-Agent': 'MyRedditApp/1.0',
                },
            }

            console.log("Fetching user information...");

            // Return a promise for each user's request
            return new Promise((resolve, reject) => {
                const req = https.request(options, (res) => {
                    let data = '';

                    res.on('data', (chunk) => {
                        data += chunk;
                    });

                    res.on('end', async () => {
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            try {
                                const response = JSON.parse(data);
                                userComments.push(response);  // Add data to the array
                                resolve();  // Resolve the promise when done
                            } catch (error) {
                                console.log(error, "Error parsing user comments");
                                reject(error);  // Reject if there's an error
                            }
                        } else {
                            reject(new Error(`Request failed with status code ${res.statusCode}`));
                        }
                    });
                });

                req.on('error', (err) => {
                    reject(err);  // Reject the promise if there’s a request error
                });

                req.end();  // End the request
            });
        });

        // Wait for all promises to resolve
        await Promise.all(promises);

        return userComments;  // Return the userComments array
    }
}

//THIS IS JUST FOR TESTING PURPOSES
(async () => {
    const subreddit = "Socionics"; // Replace with the subreddit you want to fetch
    const reddit = new userPull(subreddit);
  
    try {
      const usersComments = await reddit.getUserComments();
      console.log("users", usersComments);
    } catch (error) {
      console.error("Error:", error.message);
    }
  })();