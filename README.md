import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from scipy.interpolate import griddata

# File path
file_path = r"C:\Users\Kishore\Downloads\CSF 4.0.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Function to create and save plots
def create_and_save_plot(plot_function, filename):
    plt.figure(figsize=(12, 8))
    plot_function()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Workforce Adaptation
def workforce_adaptation():
    categories = ['Work_Experience', 'IT_Specialists', 'Industry_4.0_Priority']
    values = df[categories].mean().values
    bars = plt.bar(categories, values)
    plt.ylabel('Average Value')
    plt.title('Workforce Readiness for Industry 4.0', fontsize=16)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')

# 2. Organizational Change
def organizational_change():
    sns.set_style("whitegrid")
    sns.lineplot(x='Organization_Size', y='Industry_4.0_Priority', data=df, label='Industry 4.0 Priority', marker='o')
    sns.lineplot(x='Organization_Size', y='Leadership', data=df, label='Leadership Readiness', marker='s')
    plt.xlabel('Organization Size', fontsize=12)
    plt.ylabel('Readiness Factors', fontsize=12)
    plt.title('Industry 4.0 Readiness Factors by Organization Size', fontsize=16)
    plt.legend(fontsize=10)
    ax2 = plt.twinx()
    sns.lineplot(x='Organization_Size', y='IT_Specialists', data=df, ax=ax2, color='red', label='IT Specialists', marker='^')
    ax2.set_ylabel('Number of IT Specialists', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    plt.xticks(rotation=45)

# 3. Financial Investment
def financial_investment():
    df_sorted = df.sort_values('Industry Type')
    industries = df_sorted['Industry Type'].unique()
    finances = [df_sorted[df_sorted['Industry Type'] == industry]['Finance'].sum() for industry in industries]
    total_finance = sum(finances)
    percentages = [finance / total_finance * 100 for finance in finances]
    layers = np.zeros((len(industries), len(industries)))
    for i in range(len(industries)):
        layers[i, i:] = percentages[i]
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(industries)))
    plt.stackplot(range(len(industries)), layers, labels=industries, baseline='sym', colors=colors)
    plt.xlabel('Industry Types', fontsize=12)
    plt.ylabel('Percentage of Financial Investment', fontsize=12)
    plt.title('Stream Graph: Financial Investment in Industry 4.0 by Industry Type', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, industry in enumerate(industries):
        y = layers[:, i].sum()
        plt.text(i, y, f'{percentages[i]:.1f}%', ha='center', va='bottom')

# 4. Data Security
def data_security():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    sns.kdeplot(data=df, x='Work_Experience', ax=ax1, shade=True)
    sns.rugplot(data=df, x='Work_Experience', ax=ax1)
    ax1.set_xlabel('Work Experience')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Plot of Work Experience')
    sns.kdeplot(data=df, x='IT_Specialists', y='Industry_4.0_Priority', cmap="YlGnBu", shade=True, cbar=True, ax=ax2)
    ax2.set_xlabel('Number of IT Specialists')
    ax2.set_ylabel('Industry 4.0 Priority')
    ax2.set_title('2D Density Plot of IT Specialists vs Industry 4.0 Priority')

# 5. Supply Chain Integration
def supply_chain_integration():
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    industry_types = df['Industry Type'].value_counts()
    angles = np.linspace(0, 2*np.pi, len(industry_types), endpoint=False)
    bars = ax.bar(angles, industry_types.values, width=0.5, bottom=0.0, alpha=0.8)
    ax.set_xticks(angles)
    ax.set_xticklabels(industry_types.index)
    plt.title('Distribution of Industry Types (Radial Histogram)', fontsize=16)
    for angle, radius, label in zip(angles, industry_types.values, industry_types.values):
        ax.text(angle, radius, str(label), ha='center', va='center')

# 6. Sustainability Improvements
def sustainability_improvements():
    sustainability_cols = ['Benefits_of_Industry_4.0_in_Sustainability', 'Sensor_Monitoring', 
                           'Harmful_Emission', 'Waste_Heat_Reduction', 'Own_Source_Energy']
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    np.random.seed(0)
    df['x'] = np.random.uniform(0, 100, len(df))
    df['y'] = np.random.uniform(0, 100, len(df))
    df['Sustainability_Score'] = df[sustainability_cols].mean(axis=1)
    Z = griddata((df['x'], df['y']), df['Sustainability_Score'], (X, Y), method='cubic')
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour).set_label('Sustainability Score', rotation=270, labelpad=15)
    plt.scatter(df['x'], df['y'], c=df['Sustainability_Score'], 
                cmap='viridis', edgecolor='black', linewidth=1, s=50)
    plt.xlabel('X Coordinate (Arbitrary)')
    plt.ylabel('Y Coordinate (Arbitrary)')
    plt.title('Isoline Map of Sustainability Scores')

# 7. Competitive Advantage
def competitive_advantage():
    df_sorted = df.sort_values('Industry_4.0_Priority', ascending=False)
    top_10 = df_sorted.head(10)
    layers = np.zeros((len(top_10), len(top_10)))
    for i in range(len(top_10)):
        layers[i, i:] = top_10['Industry_4.0_Priority'].iloc[i]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
    plt.stackplot(range(len(top_10)), layers, labels=top_10.index, colors=colors)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Industry 4.0 Priority', fontsize=12)
    plt.title('Sorted Stream Graph: Top 10 Countries by Industry 4.0 Priority', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Countries')

# 8. Operational Efficiency
def operational_efficiency():
    grouped_df = df.groupby('Industry Type')[['Sensor_Monitoring', 'Sustainability_Project_Management']].mean().reset_index()
    fig = px.sunburst(
        grouped_df, 
        path=['Industry Type'], 
        values='Sensor_Monitoring',
        color='Sustainability_Project_Management',
        hover_data=['Sensor_Monitoring', 'Sustainability_Project_Management'],
        color_continuous_scale='RdBu',
        title='Industry Types by Sensor Monitoring and Sustainability Project Management'
    )
    fig.update_layout(width=800, height=800, title_font_size=20, title_x=0.5)
    fig.show()

# 9. Predictive Maintenance
def predictive_maintenance():
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=df, x='Work_Experience', y='Sensor_Monitoring', 
                cmap="viridis", shade=True, cbar=True)
    plt.xlabel('Work Experience', fontsize=12)
    plt.ylabel('Sensor Monitoring Score', fontsize=12)
    plt.title('Density Plot: Work Experience vs Sensor Monitoring Score', fontsize=16)
    plt.colorbar().set_label('Density', rotation=270, labelpad=15)
    sns.scatterplot(data=df, x='Work_Experience', y='Sensor_Monitoring', 
                    hue='Industry Type', palette='Set2', legend='brief')
    plt.legend(title='Industry Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# 10. Supply Chain Optimization
def supply_chain_optimization():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    industry_counts = df['Industry Type'].value_counts()
    ax1.bar(industry_counts.index, industry_counts.values)
    ax1.set_xlabel('Industry Type')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Industry Types')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    size_counts = df['Organization_Size'].value_counts()
    ax2.bar(size_counts.index, size_counts.values)
    ax2.set_xlabel('Organization Size')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Organization Sizes')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Main execution
if __name__ == "__main__":
    create_and_save_plot(workforce_adaptation, 'workforce_adaptation.png')
    create_and_save_plot(organizational_change, 'organizational_change.png')
    create_and_save_plot(financial_investment, 'financial_investment.png')
    create_and_save_plot(data_security, 'data_security.png')
    create_and_save_plot(supply_chain_integration, 'supply_chain_integration.png')
    create_and_save_plot(sustainability_improvements, 'sustainability_improvements.png')
    create_and_save_plot(competitive_advantage, 'competitive_advantage.png')
    operational_efficiency()
    create_and_save_plot(predictive_maintenance, 'predictive_maintenance.png')
    create_and_save_plot(supply_chain_optimization, 'supply_chain_optimization.png')
    print("All plots have been generated and saved.")
