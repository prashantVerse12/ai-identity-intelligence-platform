import os
import subprocess
import sys

def main():
    # Detect if we can run the full version
    try:
        import fastapi
        import uvicorn
        import sklearn
        has_deps = True
    except ImportError:
        has_deps = False

    # 1. Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # 2. Train model if possible and missing
    if has_deps:
        model_path = os.path.join("models", "career_model.joblib")
        if not os.path.exists(model_path):
            print("Training local ML model...")
            try:
                subprocess.run([sys.executable, "train_model.py"], check=True)
            except Exception as e:
                print(f"Training failed: {e}")

    # 3. Choose backend
    if has_deps:
        print("Launching Production FastAPI Backend...")
        backend_path = os.path.join("backend", "main.py")
    else:
        print("Launching compatibility Lite Backend (StdLib)...")
        backend_path = os.path.join("backend", "lite_server.py")
    
    subprocess.run([sys.executable, backend_path])

if __name__ == "__main__":
    main()
