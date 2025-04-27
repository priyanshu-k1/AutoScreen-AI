import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)
class ModelProcessor:
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        try:
            self.models = {
                "Casual+Skills": {
                    'model': self.load_model(os.path.join(model_dir, 'casualSkillsOnlyModel.pkl')),
                    'processor': self.process_casual_skills
                },
                "skills+strict": {
                    'model': self.load_model(os.path.join(model_dir, 'StrictSkillsOnlyModel.pkl')),
                    'processor': self.process_strict_skills
                },
                "full+casual": {
                    'model': self.load_model(os.path.join(model_dir, 'enhanced_resume_matcher.pkl')),
                    'processor': self.process_casual_full
                },
                "full+strict": {
                    'model': self.load_model(os.path.join(model_dir, 'enhanced_resume_matcher_Strict.pkl')),
                    'processor': self.process_strict_full
                }
            }
        except Exception as e:
            logger.info(f"Error loading models: {e}")
            logger.info(f"Error loading models:{self.models}")
            
        # Load additional resources for full profile processing
        try:
            with open(os.path.join(model_dir, "skills_vectorizer.pkl"), "rb") as f:
                self.skills_vectorizer = pickle.load(f)

            with open(os.path.join(model_dir, "feature_scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            logger.info(f"Error loading additional resources: {e}")
            self.skills_vectorizer = None
            self.scaler = None
    
    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    def select_model(self, model_type):
        """Select the processing function based on model name"""
        return self.models.get(model_type, self.models["Casual+Skills"])
    
    # From Here the actual logic starts for the processing the resume and job description
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()
    
    def process_casual_skills(self, resume_data, job_prefs):
        skills = resume_data.get('skills', [])
        job_skills = job_prefs.get('skills', [])
        
        logger.info("Using Casual+Skills model for processing")
      
        # Convert to strings for TF-IDF processing
        resume_text = self.clean_text( ' '.join(skills))
        job_text = self.clean_text( ' '.join(job_skills))
     
        # Create feature vectors
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_text, job_text])
        
        # Compute similarity
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        
        # Use the model to predict match (0 or 1)
        model = self.models["Casual+Skills"]['model']
        prediction = model.predict(pd.DataFrame([[score]], columns=['match_score']))[0]
        
        return {
            "score": round(score, 4),
            "prediction": prediction,
            "matched_skills": [s for s in skills if s in job_skills],
            "missing_skills": [s for s in job_skills if s not in skills]
        }

    def process_strict_skills(self, resume_data, job_prefs):
        skills = resume_data.get('skills', [])
        job_skills = job_prefs.get('skills', [])
        
        logger.info("Using skills+strict model for processing")
        
        
        def clean_text(text, skill_synonyms, skill_weights):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)  
            text = re.sub(r'\s+', ' ', text)     
              
            for synonym, standard in skill_synonyms.items():
                text = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, text, flags=re.IGNORECASE)
            
            # Apply skill weighting
            weighted_skills = []
            for skill, weight in skill_weights.items():
                if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                    # Add the skill multiple times based on its weight
                    repetitions = int(weight * 2)
                    weighted_skills.extend([skill] * repetitions)
            
            # If we found weighted skills, add them to the text
            if weighted_skills:
                return text.strip() + " " + " ".join(weighted_skills)
            return text.strip()
        
        model_package = self.models["skills+strict"]['model']
        
        model = model_package['model']
        vectorizer = model_package['vectorizer']
        optimal_threshold = model_package['optimal_threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold}")
        
        skill_synonyms = model_package['skill_synonyms']
        skill_weights = model_package['skill_weights']
        
        resume_clean = clean_text(''.join(skills), skill_synonyms, skill_weights)
        job_clean = clean_text(''.join(job_skills), skill_synonyms, skill_weights)
        
        
        
        try:
            # Try to transform with existing vocabulary
            resume_vector = vectorizer.transform([resume_clean])
            job_vector = vectorizer.transform([job_clean])
        except:
            # If new vocabulary is encountered, fit and transform
            combined = [resume_clean, job_clean]
            vectors = vectorizer.fit_transform(combined)
            resume_vector = vectors[0:1]
            job_vector = vectors[1:2]
        # Compute similarity score
        match_score = cosine_similarity(resume_vector, job_vector)[0][0]
        match_score_rounded = round(match_score, 4)
        
        # Get model prediction
        model_prediction = model.predict(pd.DataFrame([[match_score_rounded]], columns=['match_score']))[0]
        
        # Get threshold-based prediction
        threshold_prediction = 1 if match_score >= optimal_threshold else 0
        
        # Combine both predictions (model may have learned patterns beyond the threshold)
        final_prediction = model_prediction
        # Generate confidence percentage (rescaled to 0-100%)
        confidence = model.predict_proba(pd.DataFrame([[match_score_rounded]], columns=['match_score']))[0][1] * 100
        return {
            "score":match_score_rounded,
            "prediction": final_prediction == 1,
            "matched_skills": [s for s in skills if s in job_skills],
            "missing_skills": [s for s in job_skills if s not in skills]
        }
    def process_strict_full(self, resume_data, job_prefs):
        """Process full profile including skills, education, experience and certifications with stricter criteria"""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        logger.info("Using full+strict model for processing")
        with open(os.path.join(model_dir, "skills_vectorizer_Strict.pkl"), "rb") as f:
                self.skills_vectorizer = pickle.load(f)

        with open(os.path.join(model_dir, "feature_scaler_Strict.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
        
        if not self.skills_vectorizer or not self.scaler:
            return {"error": "Full profile processing resources not available"}
        
        threshold = 0.35  # Higher threshold for stricter selection
        
        # Extract data from inputs
        resume_skills = resume_data.get('skills', [])
        job_skills = job_prefs.get('skills', [])
        
        education = resume_data.get('education', '')
        
        experience_years = float(resume_data.get('experience', 0))
        certifications = resume_data.get('certifications', [])
        
        # Preprocess inputs
        resume_skills_text = " ".join([str(skill).lower().strip() for skill in resume_skills])
        job_skills_text = " ".join([str(skill).lower().strip() for skill in job_skills])
        
        # Calculate skill similarity
        resume_skills_vec = self.skills_vectorizer.transform([resume_skills_text])
        job_skills_vec = self.skills_vectorizer.transform([job_skills_text])
        skill_similarity = cosine_similarity(resume_skills_vec, job_skills_vec)[0][0]
        
        # Calculate education score
        degree_mapping = {
            # Doctoral Level - Highest Level (5)
            'phd': 5,
            'ph.d.': 5,
            'ph.d': 5,
            'doctorate': 5,
            'doctoral': 5,
            'doctor of philosophy': 5,
            'doctor of education': 5,
            'ed.d': 5,
            'ed.d.': 5,
            'doctor of business administration': 5,
            'dba': 5,
            'd.b.a.': 5,
            'doctor of medicine': 5,
            'md': 5,
            'm.d.': 5,
            'doctor of engineering': 5,
            'eng.d': 5,
            'doctor of science': 5,
            'sc.d': 5,
            'doctor of law': 5,
            'j.d.': 5,
            'jd': 5,
            'juris doctor': 5,
            
            # Master's Level (4)
            'master': 4,
            'masters': 4,
            'master of science': 4,
            'master of arts': 4,
            'master of business administration': 4,
            'master of engineering': 4,
            'master of education': 4,
            'master of fine arts': 4,
            'master of public administration': 4,
            'master of public health': 4,
            'master of social work': 4,
            'master of laws': 4,
            'master of architecture': 4,
            'm.s.': 4,
            'ms': 4,
            'm.a.': 4,
            'ma': 4,
            'mba': 4,
            'm.b.a.': 4,
            'm.eng': 4,
            'meng': 4,
            'med': 4,
            'm.ed': 4,
            'mfa': 4,
            'm.f.a.': 4,
            'mpa': 4,
            'mph': 4,
            'msw': 4,
            'll.m': 4,
            'llm': 4,
            'm.arch': 4,
            'march': 4,
            'postgraduate': 4,
            'post-graduate': 4,
            'graduate degree': 4,
            
            # Bachelor's Level (3)
            'bachelor': 3,
            'bachelors': 3,
            'baccalaureate': 3,
            'bachelor of science': 3,
            'bachelor of arts': 3,
            'bachelor of business administration': 3,
            'bachelor of engineering': 3,
            'bachelor of education': 3,
            'bachelor of fine arts': 3,
            'bachelor of architecture': 3,
            'bachelor of technology': 3,
            'b.s.': 3,
            'bs': 3,
            'b.a.': 3,
            'ba': 3,
            'b.b.a.': 3,
            'bba': 3,
            'b.eng': 3,
            'beng': 3,
            'b.ed': 3,
            'bed': 3,
            'bfa': 3,
            'b.f.a.': 3,
            'b.arch': 3,
            'barch': 3,
            'b.tech': 3,
            'btech': 3,
            'undergraduate': 3,
            'undergraduate degree': 3,
            
            # Associate Level (2)
            'associate': 2,
            'associates': 2,
            'associate degree': 2,
            'associate of arts': 2,
            'associate of science': 2,
            'associate of applied science': 2,
            'a.a.': 2,
            'aa': 2,
            'a.s.': 2,
            'as': 2,
            'a.a.s.': 2,
            'aas': 2,
            'foundation degree': 2,
            'technical degree': 2,
            'vocational degree': 2,
            'diploma': 2,
            'advanced diploma': 2,
            '2-year degree': 2,
            'two-year degree': 2,
            'community college': 2,
            
            # High School Level (1)
            'high school': 1,
            'high school diploma': 1,
            'high school degree': 1,
            'secondary education': 1,
            'secondary school': 1,
            'ged': 1,
            'general education diploma': 1,
            'general educational development': 1,
            'secondary diploma': 1,
            'hsd': 1,
            'hsed': 1,
            'high school equivalency diploma': 1,
            'a-levels': 1,
            'advanced levels': 1,
            
            # No Formal Education (0) - Optional addition
            'no degree': 0,
            'no formal education': 0,
            'self-taught': 0,
            'self taught': 0,
            'some college': 0.5,  # Some college but no degree
            'some university': 0.5,
            'incomplete degree': 0.5,
            'coursework': 0.5,
        }
        education_score = 0
        education = str(education).lower()
        for term, score in degree_mapping.items():
            if term in education:
                education_score = max(education_score, score)
        education_score = education_score / 5.0  # Normalize
        
        # Calculate certification count
        cert_count = len(certifications) / 10.0  # Normalize
        
        # Calculate experience score
        experience_score = min(experience_years, 20) / 20  # Normalize and cap at 20 years
        
        # Calculate keyword match
        resume_skills_set = set(resume_skills_text.split())
        job_skills_set = set(job_skills_text.split())
        keyword_match = len(resume_skills_set.intersection(job_skills_set)) / len(job_skills_set) if job_skills_set else 0
        
        # Create feature vector
        features = [[skill_similarity, education_score, cert_count, experience_score, keyword_match]]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probability
        model = self.models["full+strict"]['model']
        proba = model.predict_proba(features_scaled)[0][1]
        
        # Make prediction
        prediction = 1 if proba >= threshold else 0
        
        # Calculate match score with adjusted weights (more weight on skills for strict model)
        match_score = (
            skill_similarity * 0.40 + 
            education_score * 0.20 + 
            cert_count * 0.10 + 
            experience_score * 0.15 + 
            keyword_match * 0.15
        )
        
        # Get matched and missing skills
        matched_skills = [s for s in resume_skills if s.lower() in [js.lower() for js in job_skills]]
        missing_skills = [s for s in job_skills if s.lower() not in [rs.lower() for rs in resume_skills]]
        
        return {
            "score": float(match_score),
            "prediction": prediction,
            "probability": float(proba),
            "skill_similarity": float(skill_similarity),
            "education_score": float(education_score * 5),  # Scale back to original range
            "certification_count": int(cert_count * 10),  # Scale back to original count
            "experience_score": float(experience_score * 20),  # Scale back to years
            "keyword_match": float(keyword_match),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_certifications": [c for c in certifications if c in job_prefs.get('certifications', [])],
            "missing_certifications": [c for c in job_prefs.get('certifications', []) if c not in certifications]
        }         
    def process_casual_full(self, resume_data, job_prefs):
        """Process full profile including skills, education, experience and certifications"""
        
        logger.info("Using full+casual model for processing")
        
        if not self.skills_vectorizer or not self.scaler:
            return {"error": "Full profile processing resources not available"}
        
        threshold = 0.25
        
        # Extract data from inputs
        resume_skills = resume_data.get('skills', [])
        job_skills = job_prefs.get('skills', [])
        
        education = resume_data.get('education', '')
        
        experience_years = float(resume_data.get('experience', 0))
        certifications = resume_data.get('certifications', [])
        
        # Preprocess inputs
        resume_skills_text = " ".join([str(skill).lower().strip() for skill in resume_skills])
        job_skills_text = " ".join([str(skill).lower().strip() for skill in job_skills])
        
        
        
        # Calculate skill similarity
        resume_skills_vec = self.skills_vectorizer.transform([resume_skills_text])
        job_skills_vec = self.skills_vectorizer.transform([job_skills_text])
        skill_similarity = cosine_similarity(resume_skills_vec, job_skills_vec)[0][0]
        
        # Calculate education score
        degree_mapping = {
            # Doctoral Level - Highest Level (5)
            'phd': 5,
            'ph.d.': 5,
            'ph.d': 5,
            'doctorate': 5,
            'doctoral': 5,
            'doctor of philosophy': 5,
            'doctor of education': 5,
            'ed.d': 5,
            'ed.d.': 5,
            'doctor of business administration': 5,
            'dba': 5,
            'd.b.a.': 5,
            'doctor of medicine': 5,
            'md': 5,
            'm.d.': 5,
            'doctor of engineering': 5,
            'eng.d': 5,
            'doctor of science': 5,
            'sc.d': 5,
            'doctor of law': 5,
            'j.d.': 5,
            'jd': 5,
            'juris doctor': 5,
            
            # Master's Level (4)
            'master': 4,
            'masters': 4,
            'master of science': 4,
            'master of arts': 4,
            'master of business administration': 4,
            'master of engineering': 4,
            'master of education': 4,
            'master of fine arts': 4,
            'master of public administration': 4,
            'master of public health': 4,
            'master of social work': 4,
            'master of laws': 4,
            'master of architecture': 4,
            'm.s.': 4,
            'ms': 4,
            'm.a.': 4,
            'ma': 4,
            'mba': 4,
            'm.b.a.': 4,
            'm.eng': 4,
            'meng': 4,
            'med': 4,
            'm.ed': 4,
            'mfa': 4,
            'm.f.a.': 4,
            'mpa': 4,
            'mph': 4,
            'msw': 4,
            'll.m': 4,
            'llm': 4,
            'm.arch': 4,
            'march': 4,
            'postgraduate': 4,
            'post-graduate': 4,
            'graduate degree': 4,
            
            # Bachelor's Level (3)
            'bachelor': 3,
            'bachelors': 3,
            'baccalaureate': 3,
            'bachelor of science': 3,
            'bachelor of arts': 3,
            'bachelor of business administration': 3,
            'bachelor of engineering': 3,
            'bachelor of education': 3,
            'bachelor of fine arts': 3,
            'bachelor of architecture': 3,
            'bachelor of technology': 3,
            'b.s.': 3,
            'bs': 3,
            'b.a.': 3,
            'ba': 3,
            'b.b.a.': 3,
            'bba': 3,
            'b.eng': 3,
            'beng': 3,
            'b.ed': 3,
            'bed': 3,
            'bfa': 3,
            'b.f.a.': 3,
            'b.arch': 3,
            'barch': 3,
            'b.tech': 3,
            'btech': 3,
            'undergraduate': 3,
            'undergraduate degree': 3,
            
            # Associate Level (2)
            'associate': 2,
            'associates': 2,
            'associate degree': 2,
            'associate of arts': 2,
            'associate of science': 2,
            'associate of applied science': 2,
            'a.a.': 2,
            'aa': 2,
            'a.s.': 2,
            'as': 2,
            'a.a.s.': 2,
            'aas': 2,
            'foundation degree': 2,
            'technical degree': 2,
            'vocational degree': 2,
            'diploma': 2,
            'advanced diploma': 2,
            '2-year degree': 2,
            'two-year degree': 2,
            'community college': 2,
            
            # High School Level (1)
            'high school': 1,
            'high school diploma': 1,
            'high school degree': 1,
            'secondary education': 1,
            'secondary school': 1,
            'ged': 1,
            'general education diploma': 1,
            'general educational development': 1,
            'secondary diploma': 1,
            'hsd': 1,
            'hsed': 1,
            'high school equivalency diploma': 1,
            'a-levels': 1,
            'advanced levels': 1,
            
            # No Formal Education (0) - Optional addition
            'no degree': 0,
            'no formal education': 0,
            'self-taught': 0,
            'self taught': 0,
            'some college': 0.5,  # Some college but no degree
            'some university': 0.5,
            'incomplete degree': 0.5,
            'coursework': 0.5,
        }
        education_score = 0
        education = str(education).lower()
        for term, score in degree_mapping.items():
            if term in education:
                education_score = max(education_score, score)
        education_score = education_score / 5.0  # Normalize
        
        # Calculate certification count
        cert_count = len(certifications) / 10.0  # Normalize
        
        # Calculate experience score
        experience_score = min(experience_years, 20) / 20  # Normalize and cap at 20 years
        
        # Calculate keyword match
        resume_skills_set = set(resume_skills_text.split())
        job_skills_set = set(job_skills_text.split())
        keyword_match = len(resume_skills_set.intersection(job_skills_set)) / len(job_skills_set) if job_skills_set else 0
        
        # Create feature vector
        features = [[skill_similarity, education_score, cert_count, experience_score, keyword_match]]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probability
        model = self.models["full+casual"]['model']
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
        
        # Get matched and missing skills
        matched_skills = [s for s in resume_skills if s.lower() in [js.lower() for js in job_skills]]
        missing_skills = [s for s in job_skills if s.lower() not in [rs.lower() for rs in resume_skills]]
        
        return {
            "score": float(match_score),
            "prediction": prediction,
            "probability": float(proba),
            "skill_similarity": float(skill_similarity),
            "education_score": float(education_score * 5),  # Scale back to original range
            "certification_count": int(cert_count * 10),  # Scale back to original count
            "experience_score": float(experience_score * 20),  # Scale back to years
            "keyword_match": float(keyword_match),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_certifications": [c for c in certifications if c in job_prefs.get('certifications', [])],
            "missing_certifications": [c for c in job_prefs.get('certifications', []) if c not in certifications]
        }
    
    def process_resumes(self, resumes_data, job_prefs, model_type="Casual+Skills"):
        """
        Process resumes and job preferences using the selected model type.
        """        
        # Select the processing function
        optedModel = self.select_model(model_type)['processor']
        # Process each resume
        results = []
        for resume in resumes_data:
            result = optedModel(resume, job_prefs)
            results.append({
                "file_name": resume.get('file_name', 'unknown'),
                "score": result.get("score", 0),
                "prediction": bool(result.get("prediction", 0)),
                "probability": float(result.get("score", 0)),
                "email": resume.get('email', ''),
                "phone": resume.get('phone', ''),
                "matched_skills": result.get('matched_skills', []),
                "missing_skills": result.get('missing_skills', []),
                "matched_certifications": result.get('matched_certifications', []),
                "missing_certifications": result.get('missing_certifications', []),
                "model_used": model_type,
                "skill_similarity": result.get("skill_similarity", 0),
                "education_score": result.get("education_score", 0),
                "certification_count": result.get("certification_count", 0),
                "experience_score": result.get("experience_score", 0),
                "keyword_match": result.get("keyword_match", 0),})         
        results.sort(key=lambda x: x['score'], reverse=True)
        return results