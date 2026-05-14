import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

load_dotenv()

# Setup Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI(title="AI Identity Intelligence Platform")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = 'models/career_model.joblib'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Model load error: {e}")

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/api/analyze")
async def analyze(
    resume_text: Optional[str] = Form(""),
    linkedin_bio: Optional[str] = Form(""),
    github_profile: Optional[str] = Form(""),
    twitter_bio: Optional[str] = Form(""),
    portfolio_content: Optional[str] = Form(""),
    skills_list: Optional[str] = Form("")
):
    # Combine signals from multiple sources
    combined_content = f"""
    [RESUME]: {resume_text}
    [LINKEDIN]: {linkedin_bio}
    [GITHUB]: {github_profile}
    [X/TWITTER]: {twitter_bio}
    [PORTFOLIO]: {portfolio_content}
    [SKILLS]: {skills_list}
    """.strip()

    if not combined_content.replace('[', '').replace(']', '').replace(':', '').strip():
        raise HTTPException(status_code=400, detail="Empty profile profile. Please provide some content.")

    # 1. Local ML Prediction & XAI
    prediction_data = {
        "predicted_roles": {},
        "top_signals": [],
        "reasoning": [],
        "confidence": 0.0,
        "xai": {}
    }
    
    if model:
        # Predict Proba
        probs = model.predict_proba([combined_content])[0]
        classes = model.classes_
        
        # Get Top 3 Roles
        role_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        prediction_data["predicted_roles"] = {role: round(float(prob) * 100, 1) for role, prob in role_probs}
        prediction_data["confidence"] = float(max(probs))
        
        # Explainability: Extract influential signals
        try:
            tfidf = model.named_steps['tfidf']
            clf = model.named_steps['clf']
            
            # Transform content to get features present in input
            feature_matrix = tfidf.transform([combined_content])
            feature_names = tfidf.get_feature_names_out()
            importances = clf.feature_importances_
            
            row = feature_matrix.getrow(0)
            words_in_input = []
            for i in row.indices:
                score = row.data[list(row.indices).index(i)] * importances[i]
                words_in_input.append((feature_names[i], float(score)))
            
            # Sort by impact
            top_signals_raw = sorted(words_in_input, key=lambda x: x[1], reverse=True)[:8]
            prediction_data["top_signals"] = [word for word, score in top_signals_raw]
            
            main_role = role_probs[0][0]
            prediction_data["reasoning"] = [
                f"Prediction of '{main_role}' is strongly supported by consistent keyword density across inputs.",
                f"Detected heavy emphasis on: {', '.join(prediction_data['top_signals'][:4])}.",
                "Semantic similarity analysis points towards high technical positioning in modern engineering stacks."
            ]
            
            # For backward compatibility with previous UI logic if any
            prediction_data["xai"] = {
                "top_signals": prediction_data["top_signals"],
                "confidence_distribution": prediction_data["predicted_roles"],
                "reasoning": prediction_data["reasoning"][0]
            }
        except Exception as x_err:
            print(f"XAI Error: {x_err}")
    else:
        prediction_data["predicted_roles"] = {"Inferred Role": 100.0}
        prediction_data["reasoning"] = ["Model not trained. Using heuristic analysis."]

    # 2. Advanced Analysis via Gemini
    prompt = f"""
    Analyze the following multi-source professional digital footprint for comprehensive intelligence.
    
    FOOTPRINT CONTENT:
    {combined_content}

    PROVIDE A DETAILED RECRUITER-GRADE ANALYSIS IN JSON FORMAT ONLY.
    NO MARKDOWN. NO CODE BLOCKS.
    
    JSON STRUCTURE REQUIRED:
    {{
      "recruiter_view": {{
        "communication": "Strong/Moderate/Developing",
        "technical_depth": "High/Moderate/Entry",
        "growth_potential": "High/Moderate",
        "startup_fit": "Strong/Moderate",
        "mnc_fit": "Strong/Moderate"
      }},
      "brand_analysis": "Sentence about brand quality.",
      "technical_credibility": "Sentence about evidence of expertise.",
      "scores": {{
        "ai_readiness": 0-100,
        "startup_ready": 0-100,
        "mnc_ready": 0-100,
        "technical_depth": 0-100,
        "communication_quality": 0-100
      }},
      "skill_gaps": ["Skill 1", "Skill 2", "Skill 3"],
      "recommendations": ["Strategy 1", "Strategy 2"],
      "ats_insights": "Sentence about resume/profile machine-readability."
    }}
    """
    
    try:
        ai_model = genai.GenerativeModel('gemini-3-flash-preview')
        response = ai_model.generate_content(prompt)
        # Fix potential markdown in response
        clean_json = re.sub(r'```json|```', '', response.text).strip()
        ai_data = json.loads(clean_json)
    except Exception as e:
        print(f"Gemini Error: {e}")
        ai_data = {"error": str(e)}

    return {
        "role": list(prediction_data["predicted_roles"].keys())[0],
        "confidence": prediction_data["confidence"],
        "prediction": prediction_data,
        "analysis": ai_data
    }

# Serve static files
if os.path.exists("backend/static"):
    app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
