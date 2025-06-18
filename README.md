# 🧠 AutoScreen-AI

>  **AI-powered Resume Screening System** — Instantly Filter, Intelligently Hire.

**[AutoScreen-AI](https://github.com/priyanshu-k1/AutoScreen-AI)** is an intelligent resume screening tool that leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to analyze resumes, match them against job roles and skills, and generate structured shortlisting results. It’s designed for recruiters and hiring platforms to automate and speed up candidate evaluation.

---

##  Features

- Drag-and-drop resume uploader  
- Supports `.pdf`, `.doc`, `.docx` formats  
- Auto-extracts skills, contact info, and certifications  
- 4-level ML model screening modes (casual, strict)  
- Role- and skill-based matching  
- Confidence scoring + final prediction  
- Filter and export results (PDF & Excel)  
- Clean, responsive UI with real-time progress feedback
---

##  Tech Stack

| Layer       | Tech Used                          |
|-------------|------------------------------------|
| **Frontend**  | HTML, CSS, JavaScript             |
| **Backend**   | Python, Flask                     |
| **ML/NLP**    | scikit-learn, spaCy, pandas       |
| **Parsing**   | python-docx, regex                |
| **Export**    | FPDF, pandas Excel Writer         |

---

## AI Model Modes

AutoScreen-AI supports multiple matching modes to ensure flexibility and precision:

| Mode            | Matching Strategy                      |
|------------------|------------------------------------------|
| Casual+Skills    | TF-IDF skill similarity                  |
| Skills+Strict    | Weighted & synonym-aware matching        |
| Full+Casual      | Skills + Education + Experience          |
| Full+Strict      | All features + stricter thresholds       |

Each model outputs:
- **Match Score**
- **Shortlisting Decision**
- **Matched vs Missing Skills**

---

## System Diagrams

### 🔹 Component Diagram
![Component Diagram](https://github.com/priyanshu-k1/AutoScreen-AI/raw/main/diagram/Component%20Diagram.png)

---

### 🔹 Data Flow Diagram
![Data Flow Diagram](https://github.com/priyanshu-k1/AutoScreen-AI/raw/main/diagram/Data%20Flow%20Diagram.png)

---

### 🔹 Sequence Diagram
![Sequence Diagram](https://github.com/priyanshu-k1/AutoScreen-AI/raw/main/diagram/Sequence%20diagram.png)

---

##  How to Run Locally

```bash
# Clone this repository
git clone https://github.com/priyanshu-k1/AutoScreen-AI.git
cd AutoScreen-AI

# Setup backend
cd BackEnd
pip install -r requirements.txt
python app.py
```
---
##
*Developed with ❤️ by Priyanshu Kumar*
