from pypdf import PdfReader
import re

class ResumeParser:
    def __init__(self):
        # Define skill sets related to career domains
        self.skill_map = {
            "Data Science/AI": ["python", "machine learning", "data science", "nlp", "pandas", "numpy", "pytorch", "tensorflow", "statistics"],
            "Web Development": ["javascript", "react", "node.js", "html", "css", "django", "flask", "backend", "frontend", "web dev"],
            "Finance": ["accounting", "financial modeling", "excel", "tally", "investment", "portfolio", "audit", "taxation"],
            "Human Resources": ["recruitment", "hiring", "payroll", "labor law", "employee engagement", "onboarding"],
            "Cloud/DevOps": ["aws", "azure", "docker", "kubernetes", "cloud", "linux", "jenkins", "terraform"],
            "Marketing": ["digital marketing", "seo", "branding", "social media", "market research", "content strategy"]
        }

    def extract_text_from_pdf(self, pdf_file):
        """Extracts text from a PDF file object (BytesIO)."""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            return text
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"

    def identify_skills(self, text):
        """Matches extracted text against the skill map."""
        found_skills = []
        found_domains = []
        
        text_lower = text.lower()
        
        for domain, skills in self.skill_map.items():
            for skill in skills:
                if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                    found_skills.append(skill.capitalize())
                    if domain not in found_domains:
                        found_domains.append(domain)
        
        return list(set(found_skills)), found_domains

    def extract_features(self, text):
        """
        Extracts categorical and numerical features for form autofill.
        """
        features = {}
        text_lower = text.lower()
        
        # 1. Percentages (Look for numbers like 75%, 8.5)
        percentages = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*%', text)
        if percentages:
            # Simple heuristic: Usually sorted SSC, HSC, Degree or highest first
            vals = sorted([float(p) for p in percentages], reverse=True)
            if len(vals) >= 1: features['ssc_p'] = vals[0]
            if len(vals) >= 2: features['hsc_p'] = vals[1]
            if len(vals) >= 3: features['degree_p'] = vals[2]
            if len(vals) >= 4: features['mba_p'] = vals[3]

        # 2. Gender — store both MBA format (M/F) and Engineering format (Male/Female)
        if re.search(r'\b(mr\.|male|boy)\b', text_lower):
            features['gender'] = 'M'
            features['gender_eng'] = 'Male'
        elif re.search(r'\b(ms\.|mrs\.|female|girl)\b', text_lower):
            features['gender'] = 'F'
            features['gender_eng'] = 'Female'

        # 3. Work Experience
        if re.search(r'\b(experience|exp|years of|work|internship)\b', text_lower):
            features['workex'] = 'Yes'
        else:
            features['workex'] = 'No'

        # 4. Specialisation
        if re.search(r'\b(finance|fin|investment|accounting)\b', text_lower):
            features['specialisation'] = 'Mkt&Fin'
        elif re.search(r'\b(hr|human resource|recruitment)\b', text_lower):
            features['specialisation'] = 'Mkt&HR'

        # 5. Degree Type
        if re.search(r'\b(tech|engineering|b\.e\.|b\.tech|computer|it)\b', text_lower):
            features['degree_t'] = 'Sci&Tech'
        elif re.search(r'\b(commerce|mgmt|bba|management|b\.com)\b', text_lower):
            features['degree_t'] = 'Comm&Mgmt'

        # 6. Engineering Specific Heuristics (Projects/Courses)
        if re.search(r'\b(hackathon|winner|prize|competition|achievement)\b', text_lower):
            features['innovative_project_eng'] = 'Yes'
            features['tech_course_eng'] = 'Yes'
        elif re.search(r'\b(project|built|developed|created)\b', text_lower):
            features['innovative_project_eng'] = 'Yes'

        if re.search(r'\b(certificate|certified|course|learning path|meta|google)\b', text_lower):
            features['training_eng'] = 'Yes'
            features['tech_course_eng'] = 'Yes'

        return features

if __name__ == "__main__":
    # Test with dummy text
    parser = ResumeParser()
    test_text = "I am a Python developer with experience in machine learning and AWS."
    skills, domains = parser.identify_skills(test_text)
    print("Skills:", skills)
    print("Domains:", domains)
