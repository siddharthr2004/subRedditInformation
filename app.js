const express = require("express");
const app = express();
const path = require("path");
const promise = require("promise");
const { spawn } = require("child_process");
const sqlite3 = require('sqlite3').verbose();
const session = require('express-session');
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true
}));

app.use(express.json());
app.use(express.urlencoded({extended: false}));
//initalize the login for each user
let db = new sqlite3.Database("users.db", (err) => {
    if (err) {
        return console.log("error instantiating database");
    }
    console.log("Connected to SQL databse");
})

db.serialize(() => {
    db.run(`CREATE TABLE IF NOT EXISTS users 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
        username TEXT, 
        password TEXT)`
    );
    db.run(`CREATE TABLE IF NOT EXISTS subTensors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        userId INTEGER,
        subreddit TEXT,
        subredditTensor TEXT,
        FOREIGN KEY (userId) REFERENCES users (id)
    )`)
});

app.get("/", (req, res) => {
    const filePath = path.join(__dirname, "login.html");
    res.sendFile(filePath, (err) => {
        if (err) {
            console.log("error reading info from branch");
        }
    })
})

app.post("/welcomePage", (req, res) => {
    const {username, password} = req.body;
    const filePath = path.join(__dirname, "welcome.html");
    db.get(`SELECT * FROM users WHERE username = ?`, [username], (err, row) => {
        if (err) {
            console.log("Failed to get user from the database");
            res.send("Our systems could not find you. Register an account with us please!");
        }
        if (row) {
            if (password == row.password) {
                req.session.username = username;
                res.sendFile(filePath, (err) => {
                    if (err) {
                        console.log("sign in succesful, error serving file");
                    }
                })
            } else {
                res.send("invalid credentials");
            }
        } else {
            res.send("invalid credentials");
        }
    })
    res.sendFile(filePath, (err) => {
        if (err) {
            console.log("error reading info from / branch");
        }
    });
})

app.post("/register", (req, res) => {
    const {username, password} = req.body
    db.run(`INSERT INTO users (username, password) VALUES (?, ?)`, [username, password], (err) => {
        if (err) {
            console.log("Error inserting into database");
        } else {
            res.send("user registered succesfully");
        }
    })
})

app.post("/getSubInfo", async (req, res) => {
    const sub = req.body.subname;
    const filePath = path.join(__dirname, "viewSubInfo.ejs");
    const dirtyReturn = new Promise((resolve, reject) => {
        const pythonScript = spawn('python3', ['AI.py', sub]);
        //python script was spawned
        let stdout = " ";
        let stderr = " "
        pythonScript.stdout.on('data', (data) => {
            //test
            stdout += data;
        })
        pythonScript.stdout.on('end', () => {
            resolve(stdout);
        })
        pythonScript.stderr.on('error', (err) => {
            stderr += err;
            reject(stderr);
        })
        pythonScript.on('close', (code) => {
            if (code != 0) {
                reject(stderr);
            }
        })
    })
    try {
        const toJsonify = await (dirtyReturn);
        const cleanedString = toJsonify.trim();
        const String = `"${cleanedString}"`;
        const valsToSend = JSON.parse(String);
        //WILL NEED TO LATER RUN THIS INTO AN EJS FILE
        db.get(`SELECT id FROM users WHERE username = ?`, [req.session.username], (err, row) => {
            if (err) {
                console.log("Error pulling from the username");
                res.send("Error pulling your information. Please input information or try again");
            }
            if (row) {
                db.run(`INSERT INTO subTensors (userId, subredditTensor, sub) VALUES (?, ?)`, 
                    [req.session.username, valsToSend, sub], (err) => {
                    if (err) {
                        console.log("error inserting subreddit tensor");
                        res.send("Error adding inputted sub into our systems");
                    }
                    console.log("Succesfully inputted tensor now");
                    res.sendFile(filePath, (err) => {
                        console.log("Error serving viewSubInfo.ejs file");
                        res.send("Error serving viewSubInfo.ejs file");
                    });
                })
            }
        })
    } catch (error) {
        console.log("error getting values");
        console.log(error);
        res.status(500).send(error);
    }
})

const port = process.env.PORT || 3002;
app.listen(port, () => {
  console.log(`Node.js app listening on port ${port}`);
});
