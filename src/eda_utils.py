import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_style():
    """Sets a premium aesthetics style for plots matching the Linear design system."""
    plt.style.use('dark_background')
    sns.set_theme(style="whitegrid", rc={
        'axes.facecolor': '#050506',
        'figure.facecolor': '#050506',
        'grid.color': '#1a1a1e',
        'text.color': '#EDEDEF',
        'axes.labelcolor': '#8A8F98',
        'xtick.color': '#8A8F98',
        'ytick.color': '#8A8F98',
        'font.family': 'sans-serif',
        'axes.edgecolor': '#1a1a1e'
    })
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def plot_placement_distribution(df):
    """Plots the distribution of Placed vs Not Placed."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#5E6AD2', '#8A8F98']
    sns.countplot(x='status', data=df, palette=colors, ax=ax)
    ax.set_title('Placement Status Distribution', color='#EDEDEF', pad=20)
    ax.set_xlabel('Status', color='#8A8F98')
    ax.set_ylabel('Count', color='#8A8F98')
    return fig

def plot_salary_distribution(df):
    """Plots salary distribution for placed students."""
    fig, ax = plt.subplots(figsize=(10, 6))
    placed_df = df[df['status'] == 'Placed']
    if len(placed_df) > 0:
        sns.histplot(placed_df['salary'], kde=True, color='#5E6AD2', alpha=0.6, ax=ax)
    ax.set_title('Salary Distribution (Placed Students)', color='#EDEDEF', pad=20)
    ax.set_xlabel('Salary (INR)', color='#8A8F98')
    ax.set_ylabel('Frequency', color='#8A8F98')
    return fig

def plot_correlation_matrix(df):
    """Plots correlation heatmap for numeric columns."""
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='mako', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', color='#EDEDEF', pad=20)
    return fig

def plot_categorical_impact(df, col):
    """Plots impact of a categorical column on placement."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=col, hue='status', data=df, palette=['#5E6AD2', '#2a2a2e'], ax=ax)
    ax.set_title(f'Placement Status by {col}', color='#EDEDEF', pad=20)
    plt.xticks(rotation=0)
    ax.legend(title='Status', frameon=False)
    return fig

def plot_score_scatter(df, x_col, y_col):
    """Scatter plot between two score percentages colored by status."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, hue='status', data=df, s=120, alpha=0.8, palette=['#5E6AD2', '#ff4b4b'], ax=ax)
    ax.set_title(f'{x_col} vs {y_col} Impact', color='#EDEDEF', pad=20)
    return fig
