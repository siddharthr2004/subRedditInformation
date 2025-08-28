const express = require("express");
const app = express();
const path = require("path");
const promise = require("promise");
const { spawn } = require("child_process");
const sqlite3 = require('sqlite3').verbose();
const session = require('express-session');
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
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
    const filePath = path.join(__dirname, "../public_html", "login.html");
    res.sendFile(filePath, (err) => {
        if (err) {
            console.log("error reading info from branch", err ? err.message : err);
        }
    })
})

app.post("/welcomePage", (req, res) => {
    const {username, password} = req.body;
    const filePath = path.join(__dirname, "../public_html", "welcome.html");
    db.get(`SELECT * FROM users WHERE username = ?`, [username], (err, row) => {
        if (err) {
            console.log("Failed to get user from the database");
            res.send("Our systems could not find you. Register an account with us please!");
        }
        if (row) {
            if (password == row.password) {
                req.session.userId = row.id;
                //test
                console.log(req.session.userId);
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
    //res.sendFile(filePath, (err) => {
    //    if (err) {
    //        console.log("error reading info from / branch");
    //    }
    //});
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

app.post("/inputSubInfo", async (req, res) => {
    const sub = req.body.subname;
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
        db.run(
            `INSERT INTO subTensors (userId, subreddit, subredditTensor) VALUES (?, ?, ?)`,
            [req.session.userId, sub, valsToSend],
            (err) => {
                if (err) {
                    console.log("error inserting subreddit tensor");
                    res.send("Error adding inputted sub into our systems");
                    return;
                }
                console.log("Succesfully inputted tensor now");
                res.send("succesfully inputted tensor info! Click back to the main page for more tasks");
            }
        );
    } catch (error) {
        console.log("error getting values");
        console.log(error);
        res.status(500).send(error);
        return;
    }
})

app.post("/getSubInfo", (req, res) => {
    const sub = req.body.subname;
    console.log(sub);
    console.log(req.session.userId);
    db.get(
        `SELECT subredditTensor FROM subTensors WHERE userId = ? AND subreddit = ?`, 
        [req.session.userId, sub], 
        (err, row) => {
            if (err) {
                console.log("error pulling from subredditTensor base, error message: ", err ? err.message : err);
            }
            if (row) {
                const valsToSend = row.subredditTensor;
                console.log(valsToSend);
                res.render("viewSubInfo", {stats: valsToSend});
            }
        }
    )
})

const port = process.env.PORT || 3002;
app.listen(port, () => {
  console.log(`Node.js app listening on port ${port}`);
});
