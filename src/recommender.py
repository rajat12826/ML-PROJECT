class Recommender:
    def __init__(self):
        pass

    def recommend_path(self, student_data, resume_domains=None):
        """
        student_data: dict with keys like 'hsc_s', 'degree_t', 'specialisation', 'etest_p', 'mba_p', 'workex', 'Stream'
        resume_domains: list of strings (domains found in resume)
        """
        recommendations = []
        
        # Rule 1: Specialisation based
        spec = student_data.get('specialisation')
        if spec == 'Mkt&Fin':
            recommendations.append("Investment Banking")
            recommendations.append("Financial Analyst")
        elif spec == 'Mkt&HR':
            recommendations.append("Talent Acquisition Specialist")
            recommendations.append("HR Strategy Consultant")
            
        # Rule 2: Degree and Skills
        degree = student_data.get('degree_t')
        hsc_s = student_data.get('hsc_s')
        if degree == 'Sci&Tech' or hsc_s == 'Science':
            recommendations.append("Data Scientist / AI Engineer")
            recommendations.append("Product Manager (Tech)")
        elif degree == 'Comm&Mgmt' or hsc_s == 'Commerce':
            recommendations.append("Business Development Manager")
            recommendations.append("Relationship Manager")
            
        # Rule 3: High Employability Test Score
        etest_p = student_data.get('etest_p', 0)
        if etest_p > 85:
            recommendations.append("Strategy Consultant")
            recommendations.append("Management Trainee")
            
        # Rule 4: Work Experience
        workex = student_data.get('workex')
        if workex == 'Yes':
            recommendations.append("Operations Manager")
            recommendations.append("Project Lead")

        # Rule 5: Resume Based Boosting
        if resume_domains:
            for domain in resume_domains:
                if domain == "Data Science/AI":
                    recommendations.insert(0, "AI Research Scientist")
                elif domain == "Web Development":
                    recommendations.insert(0, "Full Stack Developer")
                elif domain == "Cloud/DevOps":
                    recommendations.insert(0, "Cloud Solutions Architect")
                elif domain == "Finance":
                    recommendations.insert(0, "Quantitative Analyst")

        # Return unique recommendations (top 3)
        # Using a list to maintain order (resume-boosted first)
        seen = set()
        unique_recs = []
        for r in recommendations:
            if r not in seen:
                unique_recs.append(r)
                seen.add(r)
                
        return unique_recs[:3]

    def get_skill_analysis(self, domain, resume_skills):
        """
        Calculates match and missing skills for a given domain.
        """
        # Hardcoded requirement map for common domains
        requirements = {
            "Data Scientist / AI Engineer": ["Python", "Machine Learning", "Statistics", "Pandas", "NLP"],
            "Full Stack Developer": ["JavaScript", "HTML", "CSS", "React", "Node.js"],
            "Comm&Mgmt": ["Financial Analyst", "Marketing Executive", "Retail Manager"],
            "Sci&Tech": ["Software Engineer", "Systems Architect", "Data Engineer"],
            "Others": ["General Manager", "Operations Lead"],
            # Engineering Specific Braches
            "Computer Science and Engineering": ["Software Engineer", "Full Stack Developer", "Systems Architect"],
            "Information Technology": ["Cloud Architect", "DevOps Engineer", "IT Consultant"],
            "Electronics and Communication Engineering": ["VLSI Design Engineer", "Embedded Systems Engineer", "Network Architect"],
            "Mechanical Engineering": ["CAD Designer", "Robotics Engineer", "Manufacturing Lead"],
            "Civil Engineering": ["Structural Engineer", "Site Manager", "Urban Planner"],
            "Electrical Engineering": ["Power Systems Engineer", "Control Systems Lead"],
            "Chemical Engineering": ["Process Engineer", "R&D Scientist"],
            "Investment Banking": ["Financial Modeling", "Excel", "Accounting", "Investment"],
            "Talent Acquisition Specialist": ["Recruitment", "Hiring", "Onboarding"],
            "Operations Manager": ["Excel", "Project Lead", "Operations"],
            "Strategy Consultant": ["Case Study", "Excel", "Market Research"]
        }
        
        # Default requirements if domain not specifically listed
        req_list = requirements.get(domain, ["Communication", "Problem Solving", "Excel"])
        
        match = [s for s in resume_skills if s in req_list]
        missing = [s for s in req_list if s not in resume_skills]
        
        return match, missing

if __name__ == "__main__":
    recommender = Recommender()
    sample_student = {
        'hsc_s': 'Science',
        'degree_t': 'Sci&Tech',
        'specialisation': 'Mkt&Fin',
        'etest_p': 90,
        'workex': 'No'
    }
    print("Recommendations for sample student:", recommender.recommend_path(sample_student))
