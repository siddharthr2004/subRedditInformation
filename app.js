const express = require("express");
const app = express();
const path = require("path");
const promise = require("promise");
const spawn = require("child_process");
const port = 3001;

app.use(express.json());
app.use(express.urlencoded({extended: false}));

app.get("/", (req, res) => {
    const filePath = path.join(__dirname, "welcome.html");
    res.sendFile(filePath, (err) => {
        if (err) {
            console.log("error reading info from / branch");
        }
    });
})

app.post("/getSubInfo", async (req, res) => {
    const sub = req.body.subName;
    const dirtyReturn = new Promise((resolve, reject) => {
        const pythonScript = spawn('python3', ['AI.py', sub]);
        let stdout = " ";
        let stderr = " "
        pythonScript.stdout.on('data', (data) => {
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
        const valsToSend = JSON.parse(toJsonify);
        //WILL NEED TO LATER RUN THIS INTO AN EJS FILE
    } catch (error) {
        console.log("error getting values");
        res.status(500).send(error);
    }
})

app.listen(port, 'localhost', (err) => {
    if (err) {
        console.log("error connecting to port", err);
    }
    console.log("listening to port 3001");
})
