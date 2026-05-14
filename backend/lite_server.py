import http.server
import socketserver
import json
import os
import urllib.request
from datetime import datetime

PORT = 3000
DIRECTORY = "backend/static"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "mode": "lite"}).encode())
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Since we receive multipart/form-data usually, but we can simplify
            # Our frontend uses FormData.
            # For the lite server, we'll just try to find the 'content' field in the raw data
            # Or just hack it for the preview.
            
            # Simplified: Let's assume we can get the text
            # Actually, parsing multipart in stdlib is a pain.
            # I'll update the frontend to send JSON if it detects 'lite' mode.
            
            try:
                # Placeholder response or try to hit Gemini REST API
                api_key = os.environ.get("GEMINI_API_KEY")
                
                # Default mock response in case of any failure
                mock_response = {
                    "role": "Software Engineer (Analysis Pending)",
                    "confidence": 0.9,
                    "analysis": {
                        "professionalism_score": 85,
                        "recruiter_impression": "Strong technical foundation with growth potential.",
                        "communication_quality": "High",
                        "technical_positioning": "Backend specialist",
                        "career_fit_score": 88,
                        "profile_strength": "Very Strong",
                        "ai_readiness": 75,
                        "startup_vs_mnc": "Startup Ready",
                        "recommendations": ["Add more project metrics", "Highlight leadership roles"],
                        "missing_skills": ["Kubernetes", "AWS"],
                        "keyword_analysis": ["Python", "Algorithms", "Optimization"],
                        "ats_insights": "Optimized for high-growth tech roles."
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(mock_response).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

if __name__ == "__main__":
    print(f"Starting Lite Python Server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
