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
        # Select all numeric-ish columns for benchmarking
        metrics = self.placed_df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [m for m in metrics if m not in ['salary', 'sl_no', 'sl no', 'slno']]
        
        benchmarks = {}
        # Avoid division by zero if no one is placed yet
        if len(self.placed_df) == 0:
            avg_placed = {m: 0 for m in metrics}
        else:
            avg_placed = self.placed_df[metrics].mean().to_dict()
        
        for m in metrics:
            student_val = student_data.get(m, 0)
            avg_val = avg_placed.get(m, 0)
            
            # Calculate percentile in the whole dataset
            if len(self.df) > 0:
                percentile = (self.df[m] < student_val).mean() * 100
            else:
                percentile = 0
            
            benchmarks[m] = {
                'student': student_val,
                'average_placed': avg_val,
                'percentile': percentile,
                'diff': student_val - avg_val
            }
            
        return benchmarks

    def get_company_tier(self, salary):
        if salary == 0: return "N/A"
        elif salary > 450000: return "Tier 1: Product / Lead roles"
        elif salary > 350000: return "Tier 2: Premium Service"
        else: return "Tier 3: Core Service"
