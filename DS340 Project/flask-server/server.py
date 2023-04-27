from flask import Flask
#"proxy": "http://127.0.0.1:5000",
app = Flask(__name__) #flask object

@app.route("/members")
def members():
    return{
        "members": ["Member1", "Member2", "Member3"]
        }

if __name__  == "__main__": #runs flask app
    app.run(debug=True)
    