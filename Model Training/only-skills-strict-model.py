import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve

# Skill mapping dictionaries for improved matching
SKILL_SYNONYMS = {
        # Programming Languages
        'js': 'javascript',
        'javascript': 'javascript',
        'py': 'python',
        'python3': 'python',
        'cpp': 'c++',
        'c++': 'c++',
        'c#': 'csharp',
        'csharp': 'csharp',
        'ts': 'typescript',
        'typescript': 'typescript',
        'rb': 'ruby',
        'ruby': 'ruby',
        'java': 'java',
        'go': 'golang',
        'golang': 'golang',
        'rust': 'rust',
        'php': 'php',
        'scala': 'scala',
        'kotlin': 'kotlin',
        'swift': 'swift',
        'perl': 'perl',
        'r': 'r-language',
        'r-lang': 'r-language',
        
        # Web Development
        'reactjs': 'react',
        'react.js': 'react',
        'react': 'react',
        'angular': 'angular',
        'angularjs': 'angular',
        'vue': 'vue',
        'vuejs': 'vue',
        'vue.js': 'vue',
        'svelte': 'svelte',
        'jquery': 'jquery',
        'nodejs': 'node',
        'node.js': 'node',
        'node': 'node',
        'express': 'express',
        'expressjs': 'express',
        'nextjs': 'next',
        'next.js': 'next',
        'gatsby': 'gatsby',
        'html': 'html',
        'html5': 'html',
        'css': 'css',
        'css3': 'css',
        'sass': 'sass',
        'scss': 'sass',
        'less': 'less',
        'tailwind': 'tailwindcss',
        'bootstrap': 'bootstrap',
        'materialui': 'material-ui',
        'material-ui': 'material-ui',
        
        # Mobile Development
        'react-native': 'react-native',
        'reactnative': 'react-native',
        'flutter': 'flutter',
        'ionic': 'ionic',
        'android': 'android',
        'ios': 'ios',
        'xamarin': 'xamarin',
        
        # Backend & DevOps
        'django': 'django',
        'flask': 'flask',
        'fastapi': 'fastapi',
        'spring': 'spring',
        'springboot': 'spring-boot',
        'spring-boot': 'spring-boot',
        'rails': 'ruby-on-rails',
        'ror': 'ruby-on-rails',
        'ruby-on-rails': 'ruby-on-rails',
        'laravel': 'laravel',
        'symfony': 'symfony',
        
        # DevOps & Cloud
        'aws': 'aws',
        'azure': 'azure',
        'gcp': 'google-cloud',
        'google-cloud': 'google-cloud',
        'docker': 'docker',
        'kubernetes': 'kubernetes',
        'k8s': 'kubernetes',
        'jenkins': 'jenkins',
        'ci/cd': 'ci-cd',
        'cicd': 'ci-cd',
        'terraform': 'terraform',
        'ansible': 'ansible',
        'githubactions': 'github-actions',
        'github-actions': 'github-actions',
        
        # Databases
        'sql': 'sql',
        'mysql': 'mysql',
        'postgresql': 'postgresql',
        'postgres': 'postgresql',
        'mongodb': 'mongodb',
        'mongo': 'mongodb',
        'nosql': 'nosql',
        'redis': 'redis',
        'firebase': 'firebase',
        'elasticsearch': 'elasticsearch',
        'cassandra': 'cassandra',
        'dynamodb': 'dynamodb',
        'oracle': 'oracle-db',
        'mssql': 'sql-server',
        'sqlserver': 'sql-server',
        
        # Data Science & Machine Learning
        'ml': 'machine-learning',
        'machine-learning': 'machine-learning',
        'ai': 'artificial-intelligence',
        'dl': 'deep-learning',
        'deep-learning': 'deep-learning',
        'nlp': 'natural-language-processing',
        'cv': 'computer-vision',
        'tensorflow': 'tensorflow',
        'tf': 'tensorflow',
        'pytorch': 'pytorch',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'scikit-learn': 'scikit-learn',
        'sklearn': 'scikit-learn',
        'keras': 'keras',
        'hadoop': 'hadoop',
        'spark': 'apache-spark',
        
        # UI/UX
        'ux': 'ux-designer',
        'ui': 'ui-designer',
        'uiux': 'ui-ux-designer',
        'ux/ui': 'ui-ux-designer',
        'ui/ux': 'ui-ux-designer',
        'figma': 'figma',
        'sketch': 'sketch',
        'adobe-xd': 'adobe-xd',
        'xd': 'adobe-xd',
        'invision': 'invision',
        'photoshop': 'photoshop',
        'ps': 'photoshop',
        'illustrator': 'illustrator',
        'ai': 'illustrator',
        
        # Testing & QA
        'qa': 'quality-assurance',
        'testing': 'software-testing',
        'junit': 'junit',
        'jest': 'jest',
        'cypress': 'cypress',
        'selenium': 'selenium',
        'testng': 'testng',
        'mocha': 'mocha',
        'chai': 'chai',
        
        # Project Management & Methodologies
        'agile': 'agile',
        'scrum': 'scrum',
        'kanban': 'kanban',
        'jira': 'jira',
        'pmp': 'project-management',
        'pm': 'project-management',
        'waterfall': 'waterfall',
        
        # Version Control
        'git': 'git',
        'github': 'github',
        'gitlab': 'gitlab',
        'bitbucket': 'bitbucket',
        'svn': 'subversion',
        
        # Communication & Soft Skills
        'communication': 'communication',
        'teamwork': 'teamwork',
        'leadership': 'leadership',
        'problem-solving': 'problem-solving',
        'critical-thinking': 'critical-thinking',
        'time-management': 'time-management',
        
        # Specialized Domains
        'blockchain': 'blockchain',
        'crypto': 'cryptocurrency',
        'iot': 'internet-of-things',
        'ar': 'augmented-reality',
        'vr': 'virtual-reality',
        'gamedev': 'game-development',
        'cybersecurity': 'cybersecurity',
        'infosec': 'information-security',
        'seo': 'seo',
        'webperf': 'web-performance',
        
        # Tools & Platforms
        'slack': 'slack',
        'atlassian': 'atlassian',
        'confluence': 'confluence',
        'wordpress': 'wordpress',
        'shopify': 'shopify',
        'salesforce': 'salesforce',
        'sap': 'sap',
        'tableau': 'tableau',
        'powerbi': 'power-bi',
        'excel': 'microsoft-excel',
        'powerpoint': 'microsoft-powerpoint',
        'word': 'microsoft-word',
        'office': 'microsoft-office',
        'vscode': 'visual-studio-code',
        'vs': 'visual-studio',
        'vim': 'vim',
        'intellij': 'intellij-idea',
        'pycharm': 'pycharm'
    }

SKILL_WEIGHTS = {
    # Programming Languages - Core
    'python': 2.0,
    'java': 2.0,
    'javascript': 1.8,
    'typescript': 1.7,
    'c++': 1.9,
    'csharp': 1.9,
    'golang': 1.8,
    'ruby': 1.7,
    'php': 1.6,
    'rust': 1.9,
    'scala': 1.7,
    'kotlin': 1.7,
    'swift': 1.8,
    'r-language': 1.7,
    'sql': 1.8,
    
    # Data Science & Machine Learning
    'machine-learning': 1.8,
    'deep-learning': 1.8,
    'artificial-intelligence': 1.7,
    'natural-language-processing': 1.7,
    'computer-vision': 1.7,
    'data-science': 1.8,
    'data-analysis': 1.6,
    'data-mining': 1.6,
    'statistics': 1.5,
    'big-data': 1.6,
    
    # ML/AI Frameworks
    'tensorflow': 1.5,
    'pytorch': 1.5,
    'keras': 1.4,
    'scikit-learn': 1.5,
    'pandas': 1.4,
    'numpy': 1.4,
    'scipy': 1.3,
    'huggingface': 1.6,
    'transformers': 1.6,
    
    # Web Development - Frontend
    'react': 1.7,
    'angular': 1.6,
    'vue': 1.6,
    'html': 1.3,
    'css': 1.3,
    'sass': 1.2,
    'tailwindcss': 1.4,
    'bootstrap': 1.3,
    'material-ui': 1.3,
    'jquery': 1.1,
    'next': 1.5,
    
    # Web Development - Backend
    'node': 1.7,
    'express': 1.5,
    'django': 1.6,
    'flask': 1.5,
    'fastapi': 1.6,
    'spring-boot': 1.6,
    'ruby-on-rails': 1.5,
    'laravel': 1.5,
    'graphql': 1.5,
    'rest-api': 1.5,
    
    # Mobile Development
    'android': 1.6,
    'ios': 1.6,
    'flutter': 1.6,
    'react-native': 1.7,
    'swift-ui': 1.5,
    'kotlin-android': 1.6,
    'xamarin': 1.4,
    
    # DevOps & Cloud
    'aws': 1.7,
    'azure': 1.6,
    'google-cloud': 1.6,
    'docker': 1.6,
    'kubernetes': 1.7,
    'ci-cd': 1.5,
    'jenkins': 1.4,
    'terraform': 1.6,
    'ansible': 1.5,
    'git': 1.4,
    'github': 1.3,
    'github-actions': 1.5,
    'devops': 1.7,
    'cloud-architecture': 1.7,
    
    # Databases
    'postgresql': 1.6,
    'mysql': 1.5,
    'mongodb': 1.5,
    'redis': 1.4,
    'elasticsearch': 1.5,
    'dynamodb': 1.5,
    'cassandra': 1.5,
    'sql-server': 1.5,
    'oracle-db': 1.5,
    'firebase': 1.4,
    'nosql': 1.4,
    
    # Data Engineering
    'apache-spark': 1.6,
    'hadoop': 1.5,
    'kafka': 1.6,
    'airflow': 1.6,
    'etl': 1.5,
    'data-warehouse': 1.6,
    'data-lake': 1.6,
    'data-engineering': 1.7,
    'data-pipeline': 1.6,
    
    # UI/UX
    'ui-designer': 1.5,
    'ux-designer': 1.5,
    'ui-ux-designer': 1.6,
    'figma': 1.4,
    'sketch': 1.3,
    'adobe-xd': 1.3,
    'user-research': 1.5,
    'wireframing': 1.4,
    'prototyping': 1.4,
    'user-testing': 1.5,
    
    # Testing & QA
    'software-testing': 1.4,
    'quality-assurance': 1.4,
    'test-automation': 1.5,
    'selenium': 1.4,
    'jest': 1.3,
    'cypress': 1.4,
    'junit': 1.3,
    'mocha': 1.3,
    'chai': 1.3,
    
    # Security
    'cybersecurity': 1.7,
    'information-security': 1.7,
    'penetration-testing': 1.6,
    'ethical-hacking': 1.6,
    'security-analysis': 1.6,
    'cryptography': 1.5,
    'network-security': 1.6,
    
    # Project Management & Methodologies
    'agile': 1.4,
    'scrum': 1.4,
    'kanban': 1.3,
    'project-management': 1.5,
    'product-management': 1.6,
    'jira': 1.2,
    'lean': 1.3,
    'six-sigma': 1.3,
    
    # Business Intelligence & Analytics
    'excel': 1.0,
    'powerbi': 1.0,
    'tableau': 1.2,
    'looker': 1.2,
    'business-intelligence': 1.4,
    'data-visualization': 1.4,
    'analytics': 1.5,
    'reporting': 1.2,
    'dashboarding': 1.3,
    
    # Domain-Specific
    'blockchain': 1.7,
    'cryptocurrency': 1.5,
    'internet-of-things': 1.6,
    'augmented-reality': 1.6,
    'virtual-reality': 1.6,
    'game-development': 1.6,
    'fintech': 1.6,
    'healthtech': 1.6,
    'edtech': 1.5,
    'e-commerce': 1.5,
    
    # Soft Skills
    'communication': 1.3,
    'teamwork': 1.2,
    'leadership': 1.4,
    'problem-solving': 1.4,
    'critical-thinking': 1.4,
    'time-management': 1.2,
    'presentation': 1.3,
    'negotiation': 1.3,
    
    # Enterprise Systems
    'salesforce': 1.4,
    'sap': 1.4,
    'oracle': 1.4,
    'microsoft-dynamics': 1.3,
    'workday': 1.3,
    'servicenow': 1.3,
    
    # Content & Marketing
    'content-strategy': 1.2,
    'seo': 1.3,
    'sem': 1.3,
    'digital-marketing': 1.4,
    'social-media-marketing': 1.3,
    'content-creation': 1.2,
    'copywriting': 1.2
}
# Apply synonym resolution
def resolve_synonyms(text):
    for synonym, standard in SKILL_SYNONYMS.items():
        # Replace whole word only using word boundaries \b
       text = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, text, flags=re.IGNORECASE)
    return text

# Apply skill weighting for improved matching
def apply_skill_weights(text):
    weighted_skills = []
    for skill, weight in SKILL_WEIGHTS.items():
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            # Add the skill multiple times based on its weight
            repetitions = int(weight * 2)  # Multiply by 2 to ensure all skills have at least 2 repetitions
            weighted_skills.extend([skill] * repetitions)
    
    # If we found weighted skills, add them to the text
    if weighted_skills:
        return text + " " + " ".join(weighted_skills)
    return text

def join_skills(skill_cell):
    try:
        skill_list = ast.literal_eval(skill_cell)
        if isinstance(skill_list, list):
            return ' '.join(skill_list)
        else:
            return str(skill_cell)
    except:
        return str(skill_cell)

# Enhanced text cleaning with synonym resolution
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Convert special chars to spaces
    text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
    text = resolve_synonyms(text)          # Apply synonym resolution
    text = apply_skill_weights(text)       # Apply skill weights
    return text.strip()

# Function to find optimal threshold
def find_optimal_threshold(y_true, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions, recalls, thresholds

# Main function to train the model with improvements
def train_resume_matching_model(data_path='resume_data.csv', output_path='StrictSkillsOnlyModel.pkl'):
    print("Loading and preparing data...")
    df = pd.read_csv(data_path)[['skills', 'skills_required']].dropna()
    
    # Apply preprocessing
    df['skills'] = df['skills'].apply(join_skills).apply(clean_text)
    df['skills_required'] = df['skills_required'].apply(join_skills).apply(clean_text)
    
    # Drop duplicates for one-to-many matching
    resumes = df['skills'].drop_duplicates().tolist()
    jobs = df['skills_required'].drop_duplicates().tolist()
    
    print(f"Processing {len(resumes)} unique resumes and {len(jobs)} unique job postings...")
    
    # Combine and vectorize
    combined = resumes + jobs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined)
    
    resume_vectors = tfidf_matrix[:len(resumes)]
    job_vectors = tfidf_matrix[len(resumes):]
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(resume_vectors, job_vectors)
    
    # Convert to a DataFrame for analysis
    similarity_df = pd.DataFrame(similarity_matrix, index=resumes, columns=jobs)
    
    # Get best job match and score per resume
    best_matches = similarity_df.idxmax(axis=1)
    best_scores = similarity_df.max(axis=1)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'resume_skills': best_matches.index,
        'best_job_match': best_matches.values,
        'match_score': best_scores.values
    })
    
    # Create a histogram of match scores
    plt.figure(figsize=(10, 6))
    plt.hist(result_df['match_score'], bins=20, edgecolor='black')
    plt.title('Distribution of Match Scores')
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find the natural threshold using k-means clustering
    from sklearn.cluster import KMeans
    scores_array = result_df['match_score'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(scores_array)
    centers = kmeans.cluster_centers_
    
    # The threshold is halfway between the two cluster centers
    natural_threshold = (centers[0][0] + centers[1][0]) / 2
    print(f"Natural threshold from clustering: {natural_threshold:.4f}")
    
    # Assign labels based on the natural threshold
    result_df['label'] = result_df['match_score'].apply(lambda x: 1 if x > natural_threshold else 0)
    
    # Split data
    X = result_df[['match_score']]
    y = result_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training logistic regression model...")
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold on test data
    optimal_threshold, precisions, recalls, thresholds = find_optimal_threshold(y_test, y_scores)
    print(f"Optimal threshold from precision-recall curve: {optimal_threshold:.4f}")
    
    # Print model performance metrics
    print("\nModel Performance:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision", color='blue')
    plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.axvline(x=natural_threshold, color='purple', linestyle='--', label=f'Natural Threshold = {natural_threshold:.4f}')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision vs Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Save the model with the optimal threshold
    model_package = {
        'model': model,
        'vectorizer': vectorizer,
        'optimal_threshold': optimal_threshold,
        'natural_threshold': natural_threshold,
        'skill_synonyms': SKILL_SYNONYMS,
        'skill_weights': SKILL_WEIGHTS
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved to {output_path}")
    print(f"Use optimal threshold of {optimal_threshold:.4f} for best precision-recall balance")
    
    return model_package

if __name__ == "__main__":
    train_resume_matching_model()