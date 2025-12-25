import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_students=200):
    """
    Generates a synthetic dataset to simulate survey results
    before the actual field data collection is completed.
    """
    data = {
        'Student_ID': np.arange(1001, 1001 + n_students),
        'Department': np.random.choice(
            ['Economics', 'Computer Eng', 'Business', 'Architecture', 'Law'], 
            n_students, 
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        'GPA': np.round(np.random.normal(3.10, 0.4, n_students), 2),
        'Weekly_AI_Hours': np.random.randint(0, 30, n_students),
        'AI_Tool_Preference': np.random.choice(
            ['ChatGPT', 'Gemini', 'Copilot', 'None'], 
            n_students
        )
    }
    
    # Introduce a slight correlation: More AI usage -> Slightly higher GPA (Hypothesis)
    df = pd.DataFrame(data)
    df['GPA'] = np.where(
        df['Weekly_AI_Hours'] > 10, 
        df['GPA'] + 0.2, 
        df['GPA']
    )
    
    # Cap GPA at 4.0 and ensure no negative values
    df['GPA'] = df['GPA'].clip(1.8, 4.0)
    
    return df

def analyze_data(df):
    """
    Performs basic exploratory data analysis (EDA).
    """
    print("--- DATA OVERVIEW ---")
    print(df.head())
    print("\n--- STATISTICS BY DEPARTMENT ---")
    print(df.groupby('Department')['Weekly_AI_Hours'].mean())
    
    # Save the simulated data
    df.to_csv('simulated_survey_data.csv', index=False)
    print("\n[INFO] Simulated data saved to 'simulated_survey_data.csv'")

def visualize_results(df):
    """
    Visualizes the relationship between AI Usage and GPA.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Weekly_AI_Hours', 
        y='GPA', 
        hue='Department', 
        palette='viridis',
        s=100, 
        alpha=0.8
    )
    
    plt.title('Correlation Analysis: AI Usage vs. Academic Performance (GPA)', fontsize=14)
    plt.xlabel('Weekly AI Usage (Hours)', fontsize=12)
    plt.ylabel('GPA (4.0 Scale)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Department')
    
    # Save the plot
    plt.savefig('analysis_results.png')
    print("[INFO] Analysis chart saved as 'analysis_results.png'")

if __name__ == "__main__":
    print("Initializing Data Simulation Pipeline...")
    student_df = generate_synthetic_data()
    analyze_data(student_df)
    visualize_results(student_df)
    print("Process Completed Successfully.")
