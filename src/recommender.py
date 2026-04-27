class Recommender:
    def __init__(self):
        pass

    def recommend_path(self, student_data):
        """
        student_data: dict with keys like 'hsc_s', 'degree_t', 'specialisation', 'etest_p', 'mba_p', 'workex', 'Stream'
        """
        recommendations = []
        
        # Rule 1: MBA Specialisation
        spec = student_data.get('specialisation')
        if spec == 'Mkt&Fin':
            recommendations.append("Investment Banking")
            recommendations.append("Financial Analyst")
        elif spec == 'Mkt&HR':
            recommendations.append("Talent Acquisition Specialist")
            recommendations.append("HR Strategy Consultant")
            
        # Rule 2: Engineering Streams
        stream = student_data.get('Stream')
        if stream == 'Computer Science and Engineering' or stream == 'Information Technology':
            recommendations.append("Full Stack Developer")
            recommendations.append("Data Scientist / AI Engineer")
            recommendations.append("Cloud Solutions Architect")
        elif stream == 'Electronics and Communication Engineering':
            recommendations.append("Embedded Systems Engineer")
            recommendations.append("VLSI Design Engineer")
        elif stream == 'Mechanical Engineering':
            recommendations.append("Robotics Engineer")
            recommendations.append("CAD Designer")
        elif stream == 'Civil Engineering':
            recommendations.append("Structural Engineer")
            recommendations.append("Site Manager")

        # Rule 3: High academic performance / Test scores
        etest_p = student_data.get('etest_p', 0)
        cgpa = student_data.get('Cgpa', 0)
        if etest_p > 85 or cgpa > 8.5:
            recommendations.append("Strategy Consultant")
            recommendations.append("Management Trainee")
            
        # Rule 4: Work Experience / Internships
        workex = student_data.get('workex')
        intern = student_data.get('Internships(Y/N)')
        if workex == 'Yes' or intern == 'Yes':
            recommendations.append("Operations Manager")
            recommendations.append("Project Lead")

        # Return unique recommendations (top 3)
        seen = set()
        unique_recs = []
        for r in recommendations:
            if r not in seen:
                unique_recs.append(r)
                seen.add(r)
                
        return unique_recs[:3]

    def get_requirements(self, domain):
        """
        Returns required skills for a given domain.
        """
        requirements = {
            "Data Scientist / AI Engineer": ["Python", "Machine Learning", "Statistics", "Pandas", "NLP"],
            "Full Stack Developer": ["JavaScript", "HTML", "CSS", "React", "Node.js"],
            "Cloud Solutions Architect": ["AWS/Azure", "Docker", "Kubernetes", "Linux"],
            "Embedded Systems Engineer": ["C/C++", "Microcontrollers", "RTOS"],
            "VLSI Design Engineer": ["Verilog/VHDL", "Digital Logic", "FPGA"],
            "Robotics Engineer": ["ROS", "Control Systems", "Python/C++"],
            "CAD Designer": ["AutoCAD", "SolidWorks", "Finite Element Analysis"],
            "Structural Engineer": ["STAAD.Pro", "Revit", "Construction Management"],
            "Investment Banking": ["Financial Modeling", "Excel", "Accounting", "Valuation"],
            "Financial Analyst": ["Excel", "SQL", "Economic Analysis"],
            "Talent Acquisition Specialist": ["Communication", "HRMS Tools", "Networking"],
            "HR Strategy Consultant": ["Organizational Behavior", "Analytics", "Strategy"],
            "Strategy Consultant": ["Case Study analysis", "Business Strategy", "Market Research"],
            "Management Trainee": ["Leadership", "Operations", "Business Communication"],
            "Operations Manager": ["Lean Six Sigma", "Supply Chain", "Project Management"],
            "Project Lead": ["Agile/Scrum", "Team Management", "MS Project"]
        }
        
        return requirements.get(domain, ["Communication", "Problem Solving", "Domain Knowledge"])

if __name__ == "__main__":
    recommender = Recommender()
    sample_student = {
        'Stream': 'Computer Science and Engineering',
        'Cgpa': 9.0,
        'Internships(Y/N)': 'Yes'
    }
    print("Recommendations for Engineering student:", recommender.recommend_path(sample_student))
