import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for professional presentation
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data(data_path):
    """Load and prepare data for EDA"""
    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic data cleaning
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('Unknown')
    
    # Handle mixed columns
    mixed_cols = [1,7,8,16,17,18,19,20,35]
    for c in mixed_cols:
        if c < len(df.columns):
            df.iloc[:, c] = df.iloc[:, c].astype(str).fillna('Unknown')
    
    print("Data cleaning completed")
    return df

def create_executive_summary(df):
    """Create executive summary statistics"""
    print("=" * 80)
    print("LOAN DEFAULT PREDICTION - EXECUTIVE SUMMARY")
    print("=" * 80)
    
    total_loans = len(df)
    default_rate = df['Default'].mean() * 100
    
    # Safe numeric calculations
    try:
        avg_income = df['Client_Income'].mean()
        avg_credit_amount = df['Credit_Amount'].mean()
    except:
        avg_income = 0
        avg_credit_amount = 0
    
    print(f"Dataset Overview:")
    print(f"   - Total Loan Applications: {total_loans:,}")
    print(f"   - Default Rate: {default_rate:.2f}%")
    print(f"   - Average Client Income: ${avg_income:,.2f}")
    print(f"   - Average Credit Amount: ${avg_credit_amount:,.2f}")
    
    return {
        'total_loans': total_loans,
        'default_rate': default_rate,
        'avg_income': avg_income,
        'avg_credit_amount': avg_credit_amount
    }

def plot_target_distribution(df):
    """Plot target variable distribution"""
    print("Creating target distribution plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    default_counts = df['Default'].value_counts()
    colors = ['#2E8B57', '#DC143C']
    ax1.bar(['No Default', 'Default'], default_counts.values, color=colors, alpha=0.8)
    ax1.set_title('Loan Default Distribution', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Loans', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for i, v in enumerate(default_counts.values):
        ax1.text(i, v + total*0.01, f'{v:,}\n({v/total*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pie chart
    ax2.pie(default_counts.values, labels=['No Default', 'Default'], 
           colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Default Rate Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Target distribution plot saved")

def plot_income_analysis(df):
    """Analyze income patterns and their relationship with defaults"""
    print("Creating income analysis plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Convert income to numeric safely
    df['Client_Income'] = pd.to_numeric(df['Client_Income'], errors='coerce')
    df['Credit_Amount'] = pd.to_numeric(df['Credit_Amount'], errors='coerce')
    
    # Remove NaN values for plotting
    df_clean = df.dropna(subset=['Client_Income', 'Credit_Amount', 'Default'])
    
    # Income distribution by default status
    df_no_default = df_clean[df_clean['Default'] == 0]['Client_Income']
    df_default = df_clean[df_clean['Default'] == 1]['Client_Income']
    
    ax1.hist(df_no_default, bins=50, alpha=0.7, label='No Default', color='#2E8B57')
    ax1.hist(df_default, bins=50, alpha=0.7, label='Default', color='#DC143C')
    ax1.set_xlabel('Client Income ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Income Distribution by Default Status', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    df_plot = df_clean[df_clean['Client_Income'] < df_clean['Client_Income'].quantile(0.95)]
    sns.boxplot(data=df_plot, x='Default', y='Client_Income', ax=ax2)
    ax2.set_title('Income Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Default Status', fontsize=12)
    ax2.set_ylabel('Client Income ($)', fontsize=12)
    
    # Income vs Credit Amount
    sample_df = df_clean.sample(n=min(5000, len(df_clean)))
    scatter = ax3.scatter(sample_df['Client_Income'], sample_df['Credit_Amount'], 
                         c=sample_df['Default'], cmap='RdYlGn', alpha=0.6)
    ax3.set_xlabel('Client Income ($)', fontsize=12)
    ax3.set_ylabel('Credit Amount ($)', fontsize=12)
    ax3.set_title('Income vs Credit Amount (Color = Default)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Default Risk')
    
    # Default rate by income quartiles
    df_clean['Income_Quartile'] = pd.qcut(df_clean['Client_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    default_by_quartile = df_clean.groupby('Income_Quartile')['Default'].mean() * 100
    
    bars = ax4.bar(default_by_quartile.index, default_by_quartile.values, 
                   color=['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1'])
    ax4.set_xlabel('Income Quartile', fontsize=12)
    ax4.set_ylabel('Default Rate (%)', fontsize=12)
    ax4.set_title('Default Rate by Income Quartile', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_income_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Income analysis plots saved")

def plot_demographic_analysis(df):
    """Analyze demographic patterns"""
    print("Creating demographic analysis plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Gender analysis
    gender_default = df.groupby('Client_Gender')['Default'].mean() * 100
    bars1 = ax1.bar(gender_default.index, gender_default.values, 
                    color=['#FF9999', '#66B2FF'])
    ax1.set_title('Default Rate by Gender', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Default Rate (%)', fontsize=12)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Education analysis
    edu_default = df.groupby('Client_Education')['Default'].mean() * 100
    edu_default = edu_default.sort_values(ascending=True)
    bars2 = ax2.barh(edu_default.index, edu_default.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(edu_default))))
    ax2.set_title('Default Rate by Education Level', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Default Rate (%)', fontsize=12)
    
    # Marital status analysis
    marital_default = df.groupby('Client_Marital_Status')['Default'].mean() * 100
    wedges, texts, autotexts = ax3.pie(marital_default.values, labels=marital_default.index, 
                                       autopct='%1.1f%%', startangle=90)
    ax3.set_title('Default Rate by Marital Status', fontsize=14, fontweight='bold')
    
    # Family size analysis
    family_default = df.groupby('Client_Family_Members')['Default'].mean() * 100
    ax4.plot(family_default.index, family_default.values, marker='o', linewidth=2, markersize=8)
    ax4.set_title('Default Rate by Family Size', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Family Members', fontsize=12)
    ax4.set_ylabel('Default Rate (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Demographic analysis plots saved")

def main():
    """Main EDA execution"""
    print("Starting Comprehensive EDA Analysis...")
    
    # Load data
    df = load_and_prepare_data('Data/Dataset.csv')
    
    # Generate executive summary
    summary_stats = create_executive_summary(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_target_distribution(df)
    plot_income_analysis(df)
    plot_demographic_analysis(df)
    
    print(f"\nEDA Complete! Generated visualization files for client presentation.")
    print(f"Files saved: eda_*.png")

if __name__ == "__main__":
    main()
