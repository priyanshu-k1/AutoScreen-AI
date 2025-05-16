from sklearn.metrics.pairwise import cosine_similarity
import joblib


def predict_match(resume_skills, job_skills, education, experience_years, certifications): 
    threshold = 0.25
    
    # Load models
    try:
        model = joblib.load("enhanced_resume_matcher.pkl")
        skills_vectorizer = joblib.load("skills_vectorizer.pkl")
        scaler = joblib.load("feature_scaler.pkl")
    except:
        return {"error": "Models not found. Please train the model first."}
    
    # Preprocess inputs
    resume_skills_text = " ".join([str(skill).lower().strip() for skill in resume_skills])
    job_skills_text = " ".join([str(skill).lower().strip() for skill in job_skills])
    
    # Calculate skill similarity
    resume_skills_vec = skills_vectorizer.transform([resume_skills_text])
    job_skills_vec = skills_vectorizer.transform([job_skills_text])
    skill_similarity = cosine_similarity(resume_skills_vec, job_skills_vec)[0][0]
    
    # Calculate education score
    degree_mapping = {
        # Abbreviated version for demo
        'phd': 5, 'doctorate': 5, 'master': 4, 'bachelor': 3, 
        'associate': 2, 'high school': 1
    }
    education_score = 0
    education = str(education).lower()
    for term, score in degree_mapping.items():
        if term in education:
            education_score = max(education_score, score)
    education_score = education_score / 5.0  
    # Calculate certification count
    cert_count = len(certifications) / 10.0  
    
    # Calculate experience score
    experience_score = min(experience_years, 20) / 20 
    
    # Calculate keyword match
    resume_skills_set = set(resume_skills_text.split())
    job_skills_set = set(job_skills_text.split())
    keyword_match = len(resume_skills_set.intersection(job_skills_set)) / len(job_skills_set) if job_skills_set else 0
    
    # Create feature vector
    features = [[skill_similarity, education_score, cert_count, experience_score, keyword_match]]
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get prediction probability
    proba = model.predict_proba(features_scaled)[0][1]
    
    # Make prediction
    prediction = 1 if proba >= threshold else 0
    
    # Calculate match score with adjusted weights
    match_score = (
        skill_similarity * 0.35 + 
        education_score * 0.15 + 
        cert_count * 0.05 + 
        experience_score * 0.15 + 
        keyword_match * 0.30
    )
    
    # Return results
    return {
        "prediction": prediction,
        "probability": float(proba),
        "match_score": float(match_score),
        "threshold": threshold,
        "skill_similarity": float(skill_similarity),
        "education_score": float(education_score * 5),  # Scale back to original range
        "certification_count": int(cert_count * 10),  # Scale back to original count
        "experience_score": float(experience_score * 20),  # Scale back to years
        "keyword_match": float(keyword_match)
    }


if __name__ == "__main__":
    # Example prediction
    result = predict_match(
        resume_skills=["SQL", "Data Analysis"],
        job_skills=["Python", "Machine Learning", "Data Science"],
        education="Master of Science in Computer Science",
        experience_years=5,
        certifications=["AWS Certified", "Microsoft Certified"],
    )
    print("\nExample prediction result:")
    print(result)