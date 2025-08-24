const express = require("express");
const app = express();
const path = require("path");
const promise = require("promise");
const { spawn } = require("child_process");

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
        res.send(valsToSend);
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
