import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("-" * 60)

# Basic dataset information
print("\n1. DATASET OVERVIEW")
print("-" * 60)
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "-"*80)
print("Dataset Info:")
print("-"*80)
df.info()

# 2. SUMMARY STATISTICS
print("\n" + "="*80)
print("2. SUMMARY STATISTICS")
print("="*80)
print("\nNumerical Features:")
print(df.describe())

print("\nCategorical Features:")
print(df.describe(include=['object']))

# 3. MISSING VALUES ANALYSIS

print("\n" + "="*80)
print("3. MISSING VALUES ANALYSIS")
print("="*80)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
}).sort_values(by='Missing Count', ascending=False)
print("\n", missing_df[missing_df['Missing Count'] > 0])

# Visualize missing values
plt.figure(figsize=(10, 6))
missing_df[missing_df['Missing Count'] > 0].plot(kind='bar', y='Percentage')
plt.title('Missing Values Percentage by Feature', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Missing Percentage (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. TARGET VARIABLE ANALYSIS (Survival)
print("\n" + "="*80)
print("4. SURVIVAL ANALYSIS (TARGET VARIABLE)")
print("="*80)
survival_counts = df['Survived'].value_counts()
print("\nSurvival Distribution:")
print(survival_counts)
print(f"\nSurvival Rate: {(survival_counts[1] / len(df)) * 100:.2f}%")
print(f"Death Rate: {(survival_counts[0] / len(df)) * 100:.2f}%")

# Survival visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
survival_counts.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#51cf66'])
axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Survived (0=No, 1=Yes)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Died', 'Survived'], rotation=0)

# Pie chart
axes[1].pie(survival_counts, labels=['Died', 'Survived'], autopct='%1.1f%%',
            colors=['#ff6b6b', '#51cf66'], startangle=90)
axes[1].set_title('Survival Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('survival_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. NUMERICAL FEATURES ANALYSIS
print("\n" + "="*80)
print("5. NUMERICAL FEATURES ANALYSIS")
print("="*80)

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    
    # Add statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplots for outlier detection
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[idx], color='lightcoral')
    axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col, fontsize=10)

plt.tight_layout()
plt.savefig('numerical_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Skewness analysis
print("\nSkewness of Numerical Features:")
for col in numerical_cols:
    skewness = df[col].skew()
    print(f"{col}: {skewness:.3f}", end="")
    if abs(skewness) < 0.5:
        print(" (Fairly Symmetrical)")
    elif abs(skewness) < 1:
        print(" (Moderately Skewed)")
    else:
        print(" (Highly Skewed)")

# 6. CATEGORICAL FEATURES ANALYSIS
print("\n" + "="*80)
print("6. CATEGORICAL FEATURES ANALYSIS")
print("="*80)

categorical_cols = ['Pclass', 'Sex', 'Embarked']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, col in enumerate(categorical_cols):
    value_counts = df[col].value_counts()
    print(f"\n{col} Distribution:")
    print(value_counts)
    
    axes[idx].bar(value_counts.index.astype(str), value_counts.values, color='teal')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. CORRELATION ANALYSIS
print("\n" + "="*80)
print("7. CORRELATION ANALYSIS")
print("="*80)

# Select numerical columns for correlation
corr_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df[corr_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nKey Correlations with Survival:")
survival_corr = correlation_matrix['Survived'].sort_values(ascending=False)
print(survival_corr)

# 8. BIVARIATE ANALYSIS - Survival vs Features
print("\n" + "="*80)
print("8. BIVARIATE ANALYSIS - SURVIVAL VS FEATURES")
print("="*80)

# Survival by Pclass
print("\nSurvival by Passenger Class:")
survival_pclass = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
print(survival_pclass)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Survival by Pclass
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=axes[0, 0], color=['#ff6b6b', '#51cf66'])
axes[0, 0].set_title('Survival by Passenger Class', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Passenger Class', fontsize=10)
axes[0, 0].set_ylabel('Count', fontsize=10)
axes[0, 0].legend(['Died', 'Survived'])
axes[0, 0].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)

# Survival by Sex
print("\nSurvival by Sex:")
survival_sex = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
print(survival_sex)

pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#51cf66'])
axes[0, 1].set_title('Survival by Gender', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Gender', fontsize=10)
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].legend(['Died', 'Survived'])
axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)

# Survival by Embarked
print("\nSurvival by Embarkation Port:")
survival_embarked = pd.crosstab(df['Embarked'], df['Survived'], normalize='index') * 100
print(survival_embarked)

pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar', ax=axes[1, 0], color=['#ff6b6b', '#51cf66'])
axes[1, 0].set_title('Survival by Embarkation Port', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Embarkation Port', fontsize=10)
axes[1, 0].set_ylabel('Count', fontsize=10)
axes[1, 0].legend(['Died', 'Survived'])

# Age distribution by Survival
axes[1, 1].hist([df[df['Survived']==0]['Age'].dropna(), 
                 df[df['Survived']==1]['Age'].dropna()],
                bins=30, label=['Died', 'Survived'], color=['#ff6b6b', '#51cf66'])
axes[1, 1].set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. PAIRPLOT FOR FEATURE RELATIONSHIPS
print("\n" + "="*80)
print("9. PAIRPLOT - FEATURE RELATIONSHIPS")
print("="*80)

# Create pairplot
pairplot_data = df[['Survived', 'Pclass', 'Age', 'Fare']].dropna()
sns.pairplot(pairplot_data, hue='Survived', palette={0: '#ff6b6b', 1: '#51cf66'}, 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot - Feature Relationships', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. KEY INSIGHTS AND PATTERNS
print("\n" + "="*80)
print("10. KEY INSIGHTS AND PATTERNS")
print("="*80)

print("\n📊 SURVIVAL PATTERNS:")
print(f"1. Overall survival rate: {(df['Survived'].sum() / len(df)) * 100:.2f}%")
print(f"2. Female survival rate: {(df[df['Sex']=='female']['Survived'].sum() / len(df[df['Sex']=='female'])) * 100:.2f}%")
print(f"3. Male survival rate: {(df[df['Sex']=='male']['Survived'].sum() / len(df[df['Sex']=='male'])) * 100:.2f}%")
print(f"4. 1st Class survival rate: {(df[df['Pclass']==1]['Survived'].sum() / len(df[df['Pclass']==1])) * 100:.2f}%")
print(f"5. 3rd Class survival rate: {(df[df['Pclass']==3]['Survived'].sum() / len(df[df['Pclass']==3])) * 100:.2f}%")

print("\n📈 DATA QUALITY:")
print(f"1. Age has {df['Age'].isnull().sum()} missing values ({(df['Age'].isnull().sum()/len(df)*100):.1f}%)")
print(f"2. Cabin has {df['Cabin'].isnull().sum()} missing values ({(df['Cabin'].isnull().sum()/len(df)*100):.1f}%)")
print(f"3. Embarked has {df['Embarked'].isnull().sum()} missing values")

print("\n🔍 ANOMALIES/OUTLIERS:")
print(f"1. Fare has extreme outliers (max: ${df['Fare'].max():.2f})")
print(f"2. Age range: {df['Age'].min():.1f} to {df['Age'].max():.1f} years")

print("\n✅ CORRELATIONS:")
print("1. Survival is negatively correlated with Pclass (higher class → better survival)")
print("2. Fare is positively correlated with Survival")
print("3. Gender plays a significant role (females had much higher survival rates)")

print("\n" + "="*80)
print("EDA COMPLETE! All visualizations saved as PNG files.")
print("="*80)
