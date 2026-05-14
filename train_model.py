import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)

    # Lowercase
    text = text.lower()

    return text
print("Loading dataset...")

dataset_path = 'dataset/resumes.csv'

dataset_path = 'dataset/resumes.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

# Load CSV safely
df = pd.read_csv(
    dataset_path,
    encoding='utf-8',
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'
)

# Keep required columns only
df = df[['Resume_str', 'Category']]

# Remove null values
df.dropna(inplace=True)

# Remove broken/garbage categories
df = df[df['Category'].astype(str).str.len() > 2]

df = df[
    ~df['Category'].astype(str).str.contains(
        r'<|>|http|www|font|span|div',
        regex=True,
        case=False
    )
]

# Keep only categories with enough samples
category_counts = df['Category'].value_counts()

valid_categories = category_counts[category_counts >= 2].index

df = df[df['Category'].isin(valid_categories)]

print("\nRemaining Categories:")
print(df['Category'].value_counts())

print("\nCleaning resume text...")
df['Resume_str'] = df['Resume_str'].apply(clean_text)

# Features and labels
X = df['Resume_str']
y = df['Category']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nBuilding pipeline...")

pipeline = Pipeline([
    (
        'tfidf',
        TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
    ),
    (
        'clf',
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

print("\nEvaluating model...")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Save metrics
with open('models/metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nSaving model...")

# Save model
joblib.dump(
    pipeline,
    'models/career_model.joblib'
)

print("\nModel saved to models/career_model.joblib")

def train():
    # All code above this function should be indented and placed inside this function,
    # except for import statements and function definitions like clean_text.
    pass  # Replace this with the actual code.

if __name__ == "__main__":
    train()
