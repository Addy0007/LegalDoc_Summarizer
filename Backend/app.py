from flask import Flask, request, jsonify, render_template
from test import summarize_pdf
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf = request.files["file"]
    if pdf.filename == "":
        return jsonify({"error": "No selected file"}), 400

    temp_path = os.path.join("temp_uploaded.pdf")
    pdf.save(temp_path)

    try:
        summary = summarize_pdf(temp_path)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
