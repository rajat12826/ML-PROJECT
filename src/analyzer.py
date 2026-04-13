import pandas as pd
import numpy as np

class Analyzer:
    def __init__(self, df):
        self.df = df
        self.placed_df = df[df['status'] == 'Placed']
        
    def get_benchmarks(self, student_data):
        """
        Compares student scores against the average of placed students.
        """
        # Identify numeric metrics based on current columns
        all_metrics = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', '10th marks', '12th marks', 'Cgpa']
        metrics = [m for m in all_metrics if m in self.df.columns]
        
        benchmarks = {}
        avg_placed = self.placed_df[metrics].mean().to_dict()
        
        for m in metrics:
            student_val = student_data.get(m, 0)
            avg_val = avg_placed.get(m, 0)
            
            # Calculate percentile in the whole dataset
            percentile = (self.df[m] < student_val).mean() * 100
            
            benchmarks[m] = {
                'student': student_val,
                'average_placed': avg_val,
                'percentile': percentile,
                'diff': student_val - avg_val
            }
            
        return benchmarks

    def get_company_tier(self, salary):
        """
        Maps salary to company tiers based on general industry standards.
        """
        if salary == 0:
            return "N/A"
        elif salary > 400000:
            return "Tier 1: Product Companies / High-end Consulting (Amazon, Google, MBB)"
        elif salary > 300000:
            return "Tier 2: Premium Service / Fintech (Accenture Strategy, Wells Fargo)"
        elif salary > 200000:
            return "Tier 3: Core Service / MNCs (TCS, Infosys, Wipro)"
        else:
            return "Startups / Local Firms"

    def get_improvement_tips(self, benchmarks):
        """Generates actionable tips based on benchmark differences."""
        tips = []
        for metric, data in benchmarks.items():
            if data['diff'] < -5:
                friendly_name = metric.replace('_p', ' %').upper()
                tips.append(f"Your {friendly_name} is below the placed average. Focus on certifications to compensate.")
        return tips[:2]
