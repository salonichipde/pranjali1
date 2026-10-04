
import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    decision = str(data.get("decision", "")).strip()
    reasons = str(data.get("reasons", "")).strip()
    concerns = str(data.get("concerns", "")).strip()
    priority = str(data.get("priority", "")).strip()

    if len(decision) < 10 or not reasons:
        return jsonify({
            "error": "Please enter your decision and main reasons."
        }), 400

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "Groq API key is not configured."}), 503

    prompt = f"""
You are The Blind Spot, a critical-thinking assistant.
Do not decide for the user.
Analyze the user's reasoning fairly and respectfully.
Identify:
1. Potential blind spots
2. Assumptions to verify
3. Alternative perspectives and trade-offs
4. Critical questions
5. Evidence to gather

Distinguish facts from possibilities. Do not invent details.
End with a reflection question, not a recommendation.

Decision: {decision}
Reasons: {reasons}
Concerns: {concerns or "Not specified"}
Priority: {priority or "Not specified"}
"""

    try:
        client = Groq(api_key=api_key)
        result = client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.3-70b-versatile"
            ),
            messages=[
                {
                    "role": "system",
                    "content": "Help users think critically without deciding for them."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )

        analysis = result.choices[0].message.content
        return jsonify({"analysis": analysis})

    except Exception:
        app.logger.exception("Groq API request failed")
        return jsonify({
            "error": "AI request failed. Check the API key, model and API limits."
        }), 502


if __name__ == "__main__":
    app.run(debug=True)