import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# Function to clean text (lowercase + remove special characters)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# Function to get match score between resume and job skills
def get_match_score(resume_text, job_text):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    # Create a new TF-IDF Vectorizer for this comparison
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean, job_clean])

    # Compute similarity between resume and job description
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score, 4)  # round for neatness

# Load the saved model (.pkl file)
with open('casualSkillsOnlyModel.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 👇 Example Input: Replace with your own values
resume_skills = "Python,Machine Learning,java,java script"
job_requirements = "java,java script, Data Science"

# Get match score
match_score = get_match_score(resume_skills, job_requirements)
print(f"Match Score: {match_score}")

# Predict using the loaded model
prediction = loaded_model.predict(pd.DataFrame([[match_score]], columns=['match_score']))
# Show final result
if prediction[0] == 1:
    print("✅ Resume is a good match for the job!")
else:
    print("❌ Resume is NOT a good match for the job.")
