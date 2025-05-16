import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc, average_precision_score
from imblearn.over_sampling import RandomOverSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load and preprocess data
df = pd.read_csv("resume_data.csv").dropna(subset=['skills', 'skills_required'])

def preprocess_list_text(text):
    """Convert string representation of list to actual list and clean text"""
    try:
        items = ast.literal_eval(text) if isinstance(text, str) else text
        if isinstance(items, list):
            return ' '.join([str(item).lower().strip() for item in items])
        return str(items).lower().strip()
    except:
        return str(text).lower().strip()

def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

def calculate_experience(start_dates, end_dates):
    """Calculate total work experience in years"""
    # Check if inputs are iterable
    if not isinstance(start_dates, (list, str)) or not isinstance(end_dates, (list, str)):
        return 0  # Return 0 years of experience if dates are missing/invalid
    
    total_days = 0
    try:
        # Convert to list if string representation
        if isinstance(start_dates, str):
            start_dates = ast.literal_eval(start_dates)
        if isinstance(end_dates, str):
            end_dates = ast.literal_eval(end_dates)
            
        for start, end in zip(start_dates, end_dates):
            if isinstance(start, str) and isinstance(end, str):
                start_date = pd.to_datetime(start, errors='coerce')
                end_date = pd.to_datetime(end, errors='coerce')
                if pd.notna(start_date) and pd.notna(end_date):
                    total_days += (end_date - start_date).days
    except:
        return 0  # Return 0 if any error occurs during processing
        
    return total_days / 365  # convert to years

# Preprocess skills and required skills
df['processed_skills'] = df['skills'].apply(preprocess_list_text).apply(clean_text)
df['processed_req_skills'] = df['skills_required'].apply(preprocess_list_text).apply(clean_text)

# Preprocess certifications
df['processed_certs'] = df['certification_skills'].apply(preprocess_list_text).apply(clean_text)

# Calculate work experience
df['experience_years'] = df.apply(
    lambda x: calculate_experience(x['start_dates'], x['end_dates']), 
    axis=1
)

# Feature Engineering
def extract_features(df):
    """Extract multiple features for matching"""
    
    # Calculate cosine similarity between skills and required skills
    skills_vectorizer = TfidfVectorizer()
    skills_matrix = skills_vectorizer.fit_transform(df['processed_skills'])
    req_skills_matrix = skills_vectorizer.transform(df['processed_req_skills'])
    skill_similarity = cosine_similarity(skills_matrix, req_skills_matrix).diagonal()
    
    # Education level scoring
    degree_mapping = {
        # Doctoral Level (Score: 5)
        'phd': 5, 'doctor': 5, 'doctorate': 5, 'dba': 5, 'md': 5, 'jd': 5, 'edd': 5, 
        'd.phil': 5, 'ph.d': 5, 'doctoral': 5, 'dphil': 5, 'dsci': 5, 'sc.d': 5, 's.j.d': 5,
        'engd': 5, 'deng': 5, 'dsc': 5, 'd.sc': 5, 'dlit': 5, 'd lit': 5, 'dha': 5,
        
        # Master's Level (Score: 4)
        'master': 4, 'masters': 4, 'mba': 4, 'ms': 4, 'm.s': 4, 'm.sc': 4, 'm.tech': 4, 
        'ma': 4, 'm.a': 4, 'mfa': 4, 'm.f.a': 4, 'msc': 4, 'meng': 4, 'm.eng': 4,
        'mphil': 4, 'm.phil': 4, 'mres': 4, 'm.res': 4, 'llm': 4, 'l.l.m': 4, 
        'med': 4, 'm.ed': 4, 'msw': 4, 'm.s.w': 4, 'mph': 4, 'm.p.h': 4, 
        'mpa': 4, 'm.p.a': 4, 'emba': 4, 'pgdip': 4, 'pg diploma': 4, 'post graduate diploma': 4,
        
        # Bachelor's Level (Score: 3)
        'bachelor': 3, 'bachelors': 3, 'bs': 3, 'b.s': 3, 'b.tech': 3, 'ba': 3, 'b.a': 3, 
        'b.sc': 3, 'bsc': 3, 'bfa': 3, 'b.f.a': 3, 'beng': 3, 'b.eng': 3, 'bba': 3, 'b.b.a': 3,
        'bed': 3, 'b.ed': 3, 'llb': 3, 'l.l.b': 3, 'bcom': 3, 'b.com': 3, 'bca': 3, 'b.c.a': 3,
        'barch': 3, 'b.arch': 3, 'bdes': 3, 'b.des': 3, 'undergraduate': 3, 'honours': 3,
        'graduate': 3, 'btech': 3, 'be': 3, 'b.e': 3,
        
        # Associate Level (Score: 2)
        'associate': 2, 'associates': 2, 'diploma': 2, 'hnd': 2, 'foundation degree': 2,
        'a.a': 2, 'aa': 2, 'a.s': 2, 'as': 2, 'a.a.s': 2, 'aas': 2, 'vocational': 2,
        'technical diploma': 2, 'technical certificate': 2, 'professional diploma': 2,
        'advanced diploma': 2, 'higher diploma': 2, 'apprenticeship': 2,
        
        # High School Level (Score: 1)
        'high school': 1, 'secondary': 1, 'secondary school': 1, 'high school diploma': 1,
        'ged': 1, 'hsed': 1, 'hse': 1, 'gcse': 1, 'a level': 1, 'a-level': 1, 'o level': 1,
        'higher secondary': 1, 'hsc': 1, 'intermediate': 1, '12th grade': 1, '12th': 1,
        'senior secondary': 1, 'baccalaureate': 1
    }
    
    def get_education_score(degrees):
        if not isinstance(degrees, str):
            return 0
        
        degrees = degrees.lower().strip()
        highest_score = 0
        
        for term, score in degree_mapping.items():
            # Check if the term appears as a whole word
            if re.search(r'\b' + re.escape(term) + r'\b', degrees):
                highest_score = max(highest_score, score)
        
        return highest_score
    
    education_scores = df['degree_names'].apply(get_education_score)
    
    # Certification count
    cert_counts = df['certification_skills'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else 0
    )
    
    # Experience score (capped at 20 years)
    experience_scores = np.minimum(df['experience_years'], 20) / 20
    
    # Calculate keyword match percentage between skills and required skills
    def calculate_keyword_match(skills, req_skills):
        if not isinstance(skills, str) or not isinstance(req_skills, str):
            return 0.0
        
        skills_set = set(skills.split())
        req_skills_set = set(req_skills.split())
        
        if not req_skills_set:
            return 0.0
            
        matches = skills_set.intersection(req_skills_set)
        return len(matches) / len(req_skills_set)
    
    keyword_match = df.apply(lambda x: calculate_keyword_match(x['processed_skills'], x['processed_req_skills']), axis=1)
    
    # Combine all features
    features = pd.DataFrame({
        'skill_similarity': skill_similarity,
        'education_score': education_scores / 5.0,  # Normalize to 0-1 range
        'certification_count': cert_counts / 10.0,  # Normalize assuming max 10 certifications
        'experience_score': experience_scores,
        'keyword_match': keyword_match
    })
    
    # Text vectorizer for other potential uses
    vectorizer = TfidfVectorizer()
    combined_text = df['processed_skills'] + ' ' + df['processed_req_skills'] + ' ' + df['processed_certs']
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    return features, vectorizer, skills_vectorizer

# Extract features
features, vectorizer, skills_vectorizer = extract_features(df)

# Print feature statistics
print("Feature statistics:")
print(features.describe())

# Create match score with adjusted weights based on correlation analysis
# Giving more weight to features that showed stronger correlation with matching
features['match_score'] = (
    features['skill_similarity'] * 0.35 + 
    features['education_score'] * 0.15 + 
    features['certification_count'] * 0.05 + 
    features['experience_score'] * 0.15 + 
    features['keyword_match'] * 0.30
)

MATCH_THRESHOLD = 0.28  
print(f"\nUsing match threshold: {MATCH_THRESHOLD}")

# Try different thresholds to see class distribution
print("\nClass distribution at different thresholds:")
for threshold in [0.15, 0.2, 0.25, 0.28, 0.3, 0.4]:  # Added 0.28 to the list
    label = features['match_score'].apply(lambda x: 1 if x > threshold else 0)
    pos_count = sum(label == 1)
    total = len(label)
    print(f"Threshold {threshold}: {pos_count} positives out of {total} ({pos_count/total*100:.2f}%)")

# Set final threshold
features['label'] = features['match_score'].apply(lambda x: 1 if x > MATCH_THRESHOLD else 0)

# Check class distribution for final threshold
print("\nFinal class distribution:")
class_counts = features['label'].value_counts()
print(class_counts)
print(f"Percentage of class 1: {features['label'].mean() * 100:.2f}%")

# Examine a sample of positive examples (not just when count < 10)
positive_indices = features.index[features['label'] == 1].tolist()
positive_count = len(positive_indices)

if positive_count > 0:
    print(f"\nExamining sample of positive examples (showing up to 5 out of {positive_count}):")
    sample_size = min(5, positive_count)
    sample_indices = np.random.choice(positive_indices, sample_size, replace=False) if positive_count > 5 else positive_indices
    
    for idx in sample_indices:
        # Make sure the index exists in both DataFrames
        if idx in df.index and idx in features.index:
            row = df.loc[idx]
            print(f"\nPositive example {idx}:")
            print(f"Skills: {row['processed_skills'][:100]}...")
            print(f"Required skills: {row['processed_req_skills'][:100]}...")
            print(f"Education: {row['degree_names']}")
            print(f"Experience years: {row['experience_years']:.1f}")
            print(f"Match score: {features.loc[idx, 'match_score']:.4f}")

# Check correlation between features
correlation_matrix = features[['skill_similarity', 'education_score', 'certification_count', 
                              'experience_score', 'keyword_match', 'match_score']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show('correlation_matrix.png')
plt.close()

# Plot distribution of match scores
plt.figure(figsize=(12, 6))
sns.histplot(features['match_score'], bins=50, kde=True)
plt.axvline(x=MATCH_THRESHOLD, color='red', linestyle='--', 
           label=f'Threshold ({MATCH_THRESHOLD})')
plt.title('Distribution of Match Scores')
plt.xlabel('Match Score')
plt.ylabel('Count')
plt.legend()
plt.show('match_score_distribution.png')
plt.close()

# Feature relationships
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.scatter(features['skill_similarity'], features['match_score'], alpha=0.3)
plt.xlabel('Skill Similarity')
plt.ylabel('Match Score')
plt.title('Skill Similarity vs Match Score')

plt.subplot(2, 2, 2)
plt.scatter(features['education_score'], features['match_score'], alpha=0.3)
plt.xlabel('Education Score')
plt.ylabel('Match Score')
plt.title('Education Score vs Match Score')

plt.subplot(2, 2, 3)
plt.scatter(features['experience_score'], features['match_score'], alpha=0.3)
plt.xlabel('Experience Score')
plt.ylabel('Match Score')
plt.title('Experience Score vs Match Score')

plt.subplot(2, 2, 4)
plt.scatter(features['keyword_match'], features['match_score'], alpha=0.3)
plt.xlabel('Keyword Match')
plt.ylabel('Match Score')
plt.title('Keyword Match vs Match Score')

plt.tight_layout()
plt.show('feature_relationships.png')
plt.close()

X = features[['skill_similarity', 'education_score', 'certification_count', 'experience_score', 'keyword_match']]
y = features['label']

if sum(y == 1) < 10:
    print("\nWARNING: Very few positive examples for model training. Results may not be reliable.")
    print(f"Found {sum(y == 1)} positive examples out of {len(y)} total examples.")
    
if sum(y == 1) < 2:
    print("\nERROR: Cannot build a model with fewer than 2 positive examples.")
    print("Saving match scores for manual review...")
    features.to_csv("match_scores_for_review.csv")
    print("Consider manually reviewing 'match_scores_for_review.csv' and adjusting your threshold.")
    # Don't exit, just skip model training
else:
    # We have at least 2 positive examples, proceed with model training
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply RandomOverSampler to handle class imbalance
    if sum(y_train == 1) < sum(y_train == 0):
        print("\nApplying RandomOverSampler to handle class imbalance")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train_scaled, y_train)
        print(f"Training data shape after oversampling: {X_train_balanced.shape}")
        print(f"Class distribution after oversampling: {np.bincount(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Calculate class weights
    class_weights = {0: 1, 1: (sum(y_train == 0) / max(1, sum(y_train == 1)))}
    print(f"\nClass weights: {class_weights}")

    # Train model with class weights
    model = LogisticRegression(class_weight=class_weights, C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    # If we have enough samples, perform cross-validation
    if sum(y == 1) >= 5:
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=min(5, sum(y == 1)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            LogisticRegression(class_weight=class_weights, C=1.0, max_iter=1000, random_state=42),
            X, y, cv=cv, scoring='average_precision'
        )
        print(f"Cross-validation AP scores: {cv_scores}")
        print(f"Mean AP: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Evaluate model on test set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")
    print(classification_report(y_test, y_pred))

    # Calculate Average Precision (better for imbalanced problems)
    average_precision = average_precision_score(y_test, y_proba)
    print(f"Average Precision Score: {average_precision:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    # Precision-Recall curve (more appropriate for imbalanced data)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=sum(y_test == 1)/len(y_test), color='red', linestyle='--', 
               label=f'Baseline (No Skill): {sum(y_test == 1)/len(y_test):.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    plt.close()

    # Optimal threshold analysis
    if sum(y_test == 1) >= 1:
        print("\nAnalyzing optimal threshold for predictions...")
        # Find optimal threshold for F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred_threshold = (y_proba >= threshold).astype(int)
            # Handle cases where one class disappears with high thresholds
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred_threshold)
                f1_scores.append(f1)
            except:
                f1_scores.append(0)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # MODIFIED: Make optimal threshold stricter by adding 0.03 (3%)
        # Add a buffer to make the threshold more selective
        optimal_threshold = min(1.0, optimal_threshold + 0.03)
        
        print(f"Optimal threshold for F1 score (with 3% stricter adjustment): {optimal_threshold:.4f}")
        
        # Re-evaluate with optimal threshold
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        print("\nPerformance at optimal threshold:")
        print(classification_report(y_test, y_pred_optimal))

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.coef_[0]
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    plt.close()
