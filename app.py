from flask import Flask, render_template, request, jsonify, send_file
import pickle, json, random, sqlite3, os
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import traceback

app = Flask(__name__)

# ---------------- LOAD ML MODEL ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print("ERROR loading model/vectorizer:", e)
    traceback.print_exc()
    model = None
    vectorizer = None

# ---------------- LOAD INTENTS ----------------
try:
    with open("intents.json", encoding="utf-8") as f:
        intents = json.load(f)
except Exception as e:
    print("ERROR loading intents.json:", e)
    traceback.print_exc()
    intents = {"intents": []}

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cur = conn.cursor()

# DROP old table if exists and create new one
cur.execute("DROP TABLE IF EXISTS chats")
cur.execute("""
CREATE TABLE chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    bot TEXT,
    time TEXT
)
""")
conn.commit()

# ---------------- CHATBOT LOGIC ----------------
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence for ML prediction

def get_response(msg):
    msg_lower = msg.lower().strip()

    # 1. Keyword fallback
    if any(word in msg_lower for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Hey!"])
    if any(word in msg_lower for word in ["bye", "goodbye", "see you"]):
        return random.choice(["Goodbye!", "See you later!", "Bye!"])
    if any(word in msg_lower for word in ["thank", "thanks"]):
        return random.choice(["You're welcome!", "No problem!", "My pleasure!"])

    # 2. ML prediction with confidence check
    if model and vectorizer and intents["intents"]:
        try:
            X = vectorizer.transform([msg])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                confidence = probs.max()
                intent = model.classes_[probs.argmax()]
                if confidence < CONFIDENCE_THRESHOLD:
                    return "Sorry, I didn't understand that."
            else:
                intent = model.predict(X)[0]

            for i in intents["intents"]:
                if i["tag"] == intent:
                    return random.choice(i["responses"])

            return "Sorry, I didn't understand that."

        except Exception as e:
            print("ERROR in ML prediction:", e)
            traceback.print_exc()
            return "Sorry, I cannot process this message right now."

    # 3. Default fallback
    return "Sorry, I didn't understand that."

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_msg = data.get("message")
        if not user_msg:
            return jsonify({"reply": "Please enter a message."}), 400

        bot_reply = get_response(user_msg)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("INSERT INTO chats (user, bot, time) VALUES (?,?,?)",
                    (user_msg, bot_reply, timestamp))
        conn.commit()

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("ERROR in /chat:", e)
        traceback.print_exc()
        return jsonify({"reply": str(e)}), 500

@app.route("/download/<filetype>")
def download(filetype):
    try:
        df = pd.read_sql("SELECT * FROM chats", conn)

        if not os.path.exists("exports"):
            os.makedirs("exports")

        if filetype == "csv":
            path = "exports/chats.csv"
            df.to_csv(path, index=False)
            return send_file(path, as_attachment=True)

        elif filetype == "excel":
            path = "exports/chats.xlsx"
            df.to_excel(path, index=False)
            return send_file(path, as_attachment=True)

        elif filetype == "pdf":
            path = "exports/chats.pdf"
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)

            for i, row in df.iterrows():
                line = f"[{row['time']}] User: {row['user']} | Bot: {row['bot']}"
                pdf.multi_cell(0, 8, line)

            pdf.output(path)
            return send_file(path, as_attachment=True)

        else:
            return "Invalid file type", 400

    except Exception as e:
        print("ERROR in /download:", e)
        traceback.print_exc()
        return "Server error", 500

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
