//THIS CLASS IS IRRELEVANT FOR THE TIME BEING
const https = require('https');
const querystring = require('querystring');
const SortedMap = require('./sortedMap');
const { spawn } = require("child_process");


class redditPull {
    constructor(subreddit) {
        this.subreddit = subreddit;
    }
    //NEW COMMENT 05/2025: No reason for this method to exist,
    //this is done within the python script 
    // Function to fetch the Reddit access token
    async fetchRedditToken() {
        return new Promise((resolve, reject) => {
            // API endpoint and credentials
            const clientId = "fFo133x0B-uMP6jvADZ8bg";
            const clientSecret = "snptPSBqbm65QIwfnB-AdkUjw-hQ1A";
            const credentials = Buffer.from(`${clientId}:${clientSecret}`).toString('base64');

            // Form data
            const postData = querystring.stringify({
                scope: "submit",
                grant_type: "password",
                username: "Ancient-Opinion-4358",
                password: "1234Bharaj)#@$",
                scope: "read",
            });

            // HTTPS request options
            const options = {
                method: "POST",
                hostname: "www.reddit.com",
                path: "/api/v1/access_token",
                headers: {
                    "Authorization": `Basic ${credentials}`,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Content-Length": Buffer.byteLength(postData),
                    "User-Agent": "MyRedditApp/1.0.0 (by One_Foundation7901)",
                },
            };

            // Create the HTTPS request
            const req = https.request(options, (res) => {
                let responseBody = "";

                console.log(`HTTP Status: ${res.statusCode}`);

                // Collect data chunks
                res.on("data", (chunk) => {
                    responseBody += chunk;
                });

                // On response end
                res.on("end", () => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        try {
                            const result = JSON.parse(responseBody);
                            resolve(result.access_token); // Resolve with the access token
                        } catch (error) {
                            reject(new Error("Failed to parse response: " + error.message));
                        }
                    } else {
                        reject(new Error(`HTTP Error: ${res.statusCode}, Response: ${responseBody}`));
                    }
                });
            });

            // Handle request errors
            req.on("error", (error) => {
                reject(new Error("Request error: " + error.message));
            });

            // Write POST data and end request
            req.write(postData);
            req.end();
        });
    }

    async fetchPosts() {
      const token = await this.fetchRedditToken(); // Get the access token
      console.log(token);
      const allPosts = new SortedMap(); // Array to store all fetched posts
      const allComments = new SortedMap(); // To store comments by author and score
      const totalPosts = 1000; // Number of posts we want to fetch
      let after = null; // Pagination marker, starts as null
    
      const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    
      return new Promise((resolve, reject) => {
        // Fetch a single batch of posts
        const fetchBatch = async () => {
          // Build the URL for the request
          let path = `/r/${this.subreddit}/top.json?limit=100&t=all`; // Base path
          if (after !== null) {
            path += `&after=${after}`; // Add the "after" parameter for pagination
            console.log(`After token: ${after}`);
          }
    
          const options = {
            hostname: 'oauth.reddit.com', // Use OAuth endpoint for authenticated requests
            path: path,
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${token}`,
              'User-Agent': 'MyRedditApp/1.0',
            },
          };
    
          console.log(`Fetching from URL: https://oauth.reddit.com${path}`);
    
          // Make the HTTPS request
          const req = https.request(options, (res) => {
            let data = ''; // Variable to accumulate the response body
    
            // Collect chunks of data
            res.on('data', (chunk) => {
              data += chunk;
            });
    
            // Process the response once it’s fully received
            res.on('end', async () => {
              if (res.statusCode === 429) {
                // Rate-limited - retry after delay
                console.log('Rate-limited. Retrying...');
                const resetTime = res.headers['x-ratelimit-reset'];
                const waitTime = Math.max(0, (parseInt(resetTime) * 1000 - Date.now())); // wait until rate limit resets
                console.log(`Waiting for ${waitTime / 1000} seconds before retrying.`);
                await delay(waitTime); // Wait before retrying
                fetchBatch(); // Retry fetching the batch
              } else if (res.statusCode >= 200 && res.statusCode < 300) {
                try {
                  const response = JSON.parse(data);
    
                  // Extract posts
                  const childrenArray = response.data.children;
                  const currentBatchPosts = [];
                  const currentBatchPostIds = [];
    
                  for (const child of childrenArray) {
                    const postData = child.data;
                    currentBatchPosts.push({postUser: postData.author, postUpvote: postData.score});
                    currentBatchPostIds.push(postData.id);
                  }
    
                  currentBatchPosts.forEach(post => allPosts.insert(post.postUser, post.postUpvote));
    
                  // Update the pagination marker
                  after = response.data.after;
                  console.log(`Fetched ${allPosts.length} posts so far.`);
    
                  // Fetch comments for the current batch of posts
                  for (const id of currentBatchPostIds) {
                    await fetchComments(id); // Await to ensure sequential execution
                  }
    
                  // Check if we need to fetch more posts
                  if (allPosts.length >= totalPosts || after === null) {
                    resolve({ posts: allPosts.toArray(), comments: allComments.toArray() }); // Resolve with all posts and comments
                  } else {
                    fetchBatch(); // Fetch the next batch
                  }
                } catch (error) {
                  console.error(`Error parsing response: ${error.message}`);
                  reject(error);
                }
              } else {
                console.error(`HTTP Error: ${res.statusCode}`);
                reject(new Error(`HTTP Error: ${res.statusCode}`));
              }
            });
          });
    
          // Handle request errors
          req.on('error', (error) => {
            console.error(`Request error: ${error.message}`);
            reject(error);
          });
    
          req.end(); // End the request
        };
    
        // Fetch comments for a specific post
        const fetchComments = (postId) => {
          return new Promise((resolve, reject) => {
            const path = `/r/${this.subreddit}/comments/${postId}.json`;
    
            const options = {
              hostname: 'oauth.reddit.com',
              path: path,
              method: 'GET',
              headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'MyRedditApp/1.0',
              },
            };
    
            const req = https.request(options, (res) => {
              let data = '';
    
              res.on('data', (chunk) => {
                data += chunk;
              });
    
              res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                  try {
                    const response = JSON.parse(data);
    
                    // Extract comments
                    const comments = response[1].data.children;
                    comments.forEach((comment) => {
                      const { author, score } = comment.data;
                      if (author) {
                        allComments.insert(author, score); // Insert into SortedMap
                      }
                    });
    
                    resolve(); // Indicate completion
                  } catch (error) {
                    console.error(`Error parsing comments: ${error.message}`);
                    reject(error);
                  }
                } else {
                  console.error(`HTTP Error: ${res.statusCode}`);
                  reject(new Error(`HTTP Error: ${res.statusCode}`));
                }
              });
            });
    
            req.on('error', (error) => {
              console.error(`Request error: ${error.message}`);
              reject(error);
            });
    
            req.end();
          });
        };
    
        // Start fetching the first batch of posts
        fetchBatch();
      });
    }

    async orderData() {
      const allUsers = new SortedMap();
      const { posts, comments } = await this.fetchPosts();
    
      // Process posts
      posts.forEach((post) => {
        const key = post.postUser; // Access the correct key for user
        const value = post.postUpvote; // Access the correct value for upvotes
    
        if (allUsers.contains(key)) {
          allUsers.addVal(key, value);
        } else {
          allUsers.insert(key, value); // Insert the new user if not present
        }
      });
    
      // Process comments
      comments.forEach((comment) => {
        const key = comment.key; // Assuming comments are { key: author, value: score }
        const value = comment.value;
    
        if (allUsers.contains(key)) {
          allUsers.addVal(key, value);
        } else {
          allUsers.insert(key, value);
        }
      });
    
      // Convert sorted map to an array
      const allUsersArray = allUsers.toArray(); // Returns [[key, value], [key, value], ...]
    
      // Extract the top 500 (or less) users
      const finalUsers = [];
      if (allUsersArray.length >= 500) {
        for (let i = 0; i < 500; ++i) {
          finalUsers.push(allUsersArray[i][0]); // Add the key (user) to finalUsers
        }
      } else {
        for (let i = 0; i < allUsersArray.length; ++i) {
          finalUsers.push(allUsersArray[i][0]); // Add the key (user) to finalUsers
        }
      }
    
      return finalUsers;
    }

    async runPyScript() {
      const users = await this.orderData();
      return new Promise((resolve, reject) => {
        //initate the process of actually spawning the python script
        const python = spawn("python3", ["AI.py"]);
        //pass in the users which will be added into the python script
        python.stdin.write(JSON.stringify(users));
        //signal the end of data. Remember that stdin is used to give IN data
        python.stdin.end();
        
        let output = "";
        let errorOutput = "";
        //caputure the output from the python script
        python.stdout.on("data", (data) => {
          output += data.toString();
        });
        //capture the errors from the python script
        python.stderr.on("data", (data) => {
          errorOutput += data.toString();
        });
        //finsish when the process finishes
        python.on("close", (code) => {
          if (code == 0) {
            resolve(output.trim());
          } else {
            reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
          }
        })
      })

    } 
  
}

async function run() {
  try {
    const redditInstance = new redditPull('Socionics');  // Create a new instance of redditPull
    const topUsers = await redditInstance.runPyScript();  // Call orderData and await its result
    console.log("Top 500 Users:", topUsers);  // Log the top 500 users
  } catch (error) {
    console.error("Error running the Reddit pull:", error);  // Handle any errors
  }
}

// Run the function
run();

module.exports = redditPull;



