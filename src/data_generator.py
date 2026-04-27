import pandas as pd
import numpy as np

def expand_mba_data(file_path, target_rows=800):
    df = pd.read_csv(file_path)
    current_rows = len(df)
    if current_rows >= target_rows:
        return
    
    extra_rows = target_rows - current_rows
    new_data = []
    
    for _ in range(extra_rows):
        row = {
            'gender': np.random.choice(df['gender'].unique()),
            'ssc_p': np.random.normal(df['ssc_p'].mean(), df['ssc_p'].std()),
            'hsc_p': np.random.normal(df['hsc_p'].mean(), df['hsc_p'].std()),
            'degree_p': np.random.normal(df['degree_p'].mean(), df['degree_p'].std()),
            'workex': np.random.choice(df['workex'].unique()),
            'etest_p': np.random.normal(df['etest_p'].mean(), df['etest_p'].std()),
            'specialisation': np.random.choice(df['specialisation'].unique()),
            'mba_p': np.random.normal(df['mba_p'].mean(), df['mba_p'].std()),
        }
        
        # Simple logical rules for status
        score = (row['ssc_p'] + row['hsc_p'] + row['degree_p'] + row['etest_p']) / 4
        if row['workex'] == 'Yes': score += 10
        
        if score > 65:
            row['status'] = 'Placed'
            row['salary'] = np.random.normal(df[df['status'] == 'Placed']['salary'].mean(), 50000)
        else:
            row['status'] = 'Not Placed'
            row['salary'] = 0.0
            
        new_data.append(row)
    
    new_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    new_df.to_csv(file_path, index=False)
    print(f"Expanded MBA data to {len(new_df)} rows.")

def expand_engineering_data(file_path, target_rows=800):
    df = pd.read_csv(file_path)
    current_rows = len(df)
    if current_rows >= target_rows:
        return
    
    extra_rows = target_rows - current_rows
    new_data = []
    
    for _ in range(extra_rows):
        row = {
            'Gender': np.random.choice(df['Gender'].unique()),
            '10th marks': np.random.normal(df['10th marks'].mean(), df['10th marks'].std()),
            '12th marks': np.random.normal(df['12th marks'].mean(), df['12th marks'].std()),
            'Stream': np.random.choice(df['Stream'].unique()),
            'Cgpa': np.random.normal(df['Cgpa'].mean(), df['Cgpa'].std()),
            'Technical Score': np.random.normal(df['Technical Score'].mean(), df['Technical Score'].std()),
            'Internships(Y/N)': np.random.choice(df['Internships(Y/N)'].unique()),
            'Projects': np.random.choice(df['Projects'].unique()),
            'Backlogs': np.random.choice(df['Backlogs'].unique()),
        }
        
        # Logical rules
        score = (row['10th marks'] + row['12th marks']) / 2 + (row['Cgpa'] * 10) + (row['Technical Score'])
        if row['Internships(Y/N)'] == 'Yes': score += 20
        if row['Backlogs'] > 1: score -= 30
        
        if score > 180:
            row['status'] = 'Placed'
        else:
            row['status'] = 'Not Placed'
            
        new_data.append(row)
    
    new_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    new_df.to_csv(file_path, index=False)
    print(f"Expanded Engineering data to {len(new_df)} rows.")

if __name__ == "__main__":
    expand_mba_data('Placement_Data_Full_Class.csv')
    expand_engineering_data('Engineering.csv')
