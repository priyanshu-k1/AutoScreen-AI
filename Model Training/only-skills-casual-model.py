import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve
import joblib
import pickle
df = pd.read_csv("resume_data.csv")[['skills', 'skills_required']].dropna()


def join_skills(skill_cell):
    try:
        skill_list = ast.literal_eval(skill_cell)
        if isinstance(skill_list, list):
            return ' '.join(skill_list)
        else:
            return str(skill_cell)
    except:
        return str(skill_cell)

# Normalize text: lowercase, remove special chars
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    return text.strip()

# Apply processing
df['skills'] = df['skills'].apply(join_skills).apply(clean_text)
df['skills_required'] = df['skills_required'].apply(join_skills).apply(clean_text)

# Drop duplicates for one-to-many matching
resumes = df['skills'].drop_duplicates().tolist()
jobs = df['skills_required'].drop_duplicates().tolist()

# Combine and vectorize
combined = resumes + jobs
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined)

resume_vectors = tfidf_matrix[:len(resumes)]
job_vectors = tfidf_matrix[len(resumes):]

# Compute cosine similarity: each resume vs all job descriptions
similarity_matrix = cosine_similarity(resume_vectors, job_vectors)

# Convert to a readable DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=resumes, columns=jobs)

# Get best job match and score per resume
best_matches = similarity_df.idxmax(axis=1)
best_scores = similarity_df.max(axis=1)

# Final result
result_df = pd.DataFrame({
    'resume_skills': best_matches.index,
    'best_job_match': best_matches.values,
    'match_score': best_scores.values
})

result_df['label'] = result_df['match_score'].apply(lambda x: 1 if x > 0.18 else 0)
# Features and labels
X = result_df[['match_score']]
y = result_df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(class_weight='balanced',random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
# Classification Report
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)


y_scores = model.predict_proba(X_test)[:, 1]  # Get probabilities for class '1' (match)
# Compute precision and recall for different thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Plotting the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label="Precision", color='blue')
plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall Curve")
plt.legend()
plt.grid(True)
plt.show()


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
joblib.dump(model, "resume_match_model.pkl")
with open('casualSkillsOnlyModel.pkl', 'wb') as f:
    pickle.dump(model, f)