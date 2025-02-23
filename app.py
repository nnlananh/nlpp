from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_socketio import SocketIO, send
from werkzeug.utils import secure_filename
from DPS_2 import read_and_analyze
import json
import os
import fitz
import docx
import io
import re
import random

with open("responses.json") as f:
    dataset = json.load(f)

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET_KEY")
socketio = SocketIO(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}

WORD_LIMIT_PATTERN = r"\d+\swords"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("upload.html")

# Route for creating a bot message object
@app.route('/chat', methods=['POST'])
def chat():
    # Get user's message.
    user_message = request.form.get('message', '').strip()
    # Get the selected model.
    model = request.form.get("model", "t5-small")

    default_yes_response = {}
    other_file_response = "Sorry. I don't support this file."
    task = None

    # Search in the message whether the user request the task limitations.
    # If the re.search function returns None or error, then there aren't any word limitations
    try:
        word_limit = int(re.search(WORD_LIMIT_PATTERN, user_message).group().split(" ")[0])
        print(word_limit)
    except:
        print("No word limit found")
        word_limit = 250

    for data in dataset:
        if data["intent"]["tags"] == "summarize":
            # Create a regex string with metacharacters to match summarization purpose
            summarize_patterns = data["intent"]["patterns"][0]
            default_yes_response["summarize"] = random.choice(data["responses"]) + f"in {word_limit} words"
        elif data["intent"]["tags"] == "paraphrase":
            # Create a regex string with metacharacters to match paraphrasing purpose
            paraphrase_patterns = data["intent"]["patterns"][0]
            default_yes_response["paraphrase"] = random.choice(data["responses"]) + f"in {word_limit} words"
        elif data["intent"]["tags"] == "read":
            # Create a regex string with metacharacters for reading purpose.
            read_patterns = data["intent"]["patterns"][0]
            default_yes_response["read"] = random.choice(data["responses"])
        elif data["intent"]["tags"] == "default":
            default_no_response = random.choice(data["responses"])

    # Defining tasks for the model
    if re.search(summarize_patterns, user_message):
        task = "summarize"

    elif re.search(paraphrase_patterns, user_message):
        task = "paraphrase"

    elif re.search(read_patterns, user_message):
        task = "read"

    if task: 
        if request.files["file"]:
            file = request.files["file"]
            filename = file.filename.lower()

            if filename.endswith(".txt"):
                text = file.read().decode("utf-8")

            elif filename.endswith(".docx"):
                doc = docx.Document(io.BytesIO(file.read()))
                text = "\n".join([p.text for p in doc.paragraphs])
            
            elif filename.endswith(".pdf"):
                pdf = fitz.open(stream=file.stream, filetype="pdf")
                text = "\n".join([p.get_text() for p in pdf])

            else:
                return jsonify({"response": other_file_response}), 201
            
            if task == "summarize" or task == "paraphrase":
                analysis, metrics = read_and_analyze(text, model, task, word_limit)
                print(metrics)
                answer_text = analysis["generation"]["generated_text"]
            else:
                answer_text = text
            
            return jsonify({"response": f"{default_yes_response[task]}\n{answer_text}"}), 201
        else:
            return jsonify({"response": default_no_response}), 201
    else:
        for data in dataset:
            if any(
                re.search(data["intent"]["patterns"][i], user_message)
                for i in range(len(data["intent"]["patterns"]))
                ):
                return jsonify({"response": random.choice(data["responses"])})
        return jsonify({"response": f"{default_no_response}"}), 201

if __name__ == "__main__":
    app.run(debug=True)