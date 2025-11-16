# AI-ML-INTERNSHIP-task-2
Titanic Dataset - Exploratory Data Analysis
About This Project
This is my EDA project on the Titanic dataset where I analyzed passenger data to understand survival patterns. I used Python and various data science libraries to explore the data and create visualizations.
What I Did
I wanted to understand what factors affected survival on the Titanic. So I analyzed the dataset using statistics and created visualizations to find patterns and insights.
Dataset

Source: Titanic Dataset from Kaggle
Size: 891 passengers
Features: 12 columns including age, sex, class, fare, etc.

Tools I Used

Python 3
Pandas for data handling
NumPy for calculations
Matplotlib and Seaborn for charts

Project Files
titanic-eda/
├── Titanic-Dataset.csv
├── titanic_eda.py
├── README.md
├── INTERVIEW_ANSWERS.md
└── outputs/ (all the charts)
 Analysis Performed
1. Basic Dataset Information

Dataset shape and structure
Data types and non-null counts
First few records preview

2. Summary Statistics

Descriptive statistics for numerical features (mean, median, std, min, max)
Frequency counts for categorical features
Distribution measures

3. Missing Values Analysis

Identified missing values in Age (19.9%), Cabin (77.1%), and Embarked (0.2%)
Visualized missing data patterns
Assessed impact on analysis

4. Target Variable Analysis (Survival)

Overall survival rate: 38.4%
Distribution visualization (bar chart and pie chart)
Class imbalance assessment

5. Numerical Features Analysis

Age: Right-skewed distribution (mean: 29.7 years)
Fare: Highly right-skewed with outliers (mean: $32.20)
SibSp & Parch: Most passengers traveled alone or with small families
Histogram and boxplot visualizations
Skewness calculation and interpretation

6. Categorical Features Analysis

Pclass: 55% in 3rd class, 24% in 1st class, 21% in 2nd class
Sex: 65% male, 35% female
Embarked: 72% from Southampton (S), 19% Cherbourg (C), 9% Queenstown (Q)

7. Correlation Analysis

Created correlation matrix for numerical features
Key findings:

Negative correlation between Pclass and Survival (-0.34)
Positive correlation between Fare and Survival (0.26)
Strong negative correlation between Pclass and Fare (-0.55)



8. Bivariate Analysis

Survival by Gender: Females had 74% survival rate vs 19% for males
Survival by Class: 1st class (63%), 2nd class (47%), 3rd class (24%)
Survival by Port: Cherbourg passengers had higher survival rates
Age Distribution: Younger passengers (children) had slightly better survival

9. Feature Relationships (Pairplot)

Visualized relationships between Age, Fare, Pclass, and Survival
Identified patterns and clusters

 Key Insights
 Survival Patterns

Gender Effect: Women had 3.9x higher survival rate than men (74% vs 19%)
Class Privilege: 1st class passengers had 2.6x better survival than 3rd class
Fare Impact: Higher fare payers had better survival chances
Age Factor: Children had slightly better survival rates
Port of Embarkation: Cherbourg passengers had higher survival (55%)

Data Quality Observations

Missing Data: Age (20%), Cabin (77%), Embarked (0.2%)
Outliers: Extreme values in Fare (max: $512.33)
Class Imbalance: Only 38.4% survived (imbalanced target variable)

 Anomalies Detected

Fare has extreme outliers indicating luxury accommodations
Age ranges from 0.42 to 80 years (babies to elderly)
Some passengers traveled with large families (up to 8 relatives)

 Visualizations Generated

Missing Values Chart - Bar chart showing missing data percentage
Survival Distribution - Count plot and pie chart
Numerical Distributions - Histograms with mean/median lines
Boxplots - Outlier detection for numerical features
Categorical Distributions - Bar charts for Pclass, Sex, Embarked
Correlation Heatmap - Correlation matrix visualization
Bivariate Analysis - Survival vs various features
Pairplot - Relationships between multiple features

 How to Run
Prerequisites
bashpip install pandas numpy matplotlib seaborn
Execution
bashpython titanic_eda.py
Output

Console displays comprehensive statistical analysis
PNG files saved in the project directory with all visualizations

 What I Learned

Data Visualization: Creating effective visualizations to communicate insights
Descriptive Statistics: Understanding central tendency and dispersion
Pattern Recognition: Identifying trends and relationships in data
Missing Data Handling: Assessing impact of missing values
Feature Relationships: Understanding correlation and multicollinearity
Outlier Detection: Using boxplots and statistical methods
Bivariate Analysis: Exploring relationships between features

Conclusions
The Titanic disaster survival was heavily influenced by:

Social-economic status (class and fare)
Gender (women-and-children-first policy)
Age (priority for children)
Location (proximity to lifeboats)

The "women and children first" evacuation protocol and class-based cabin locations played critical roles in survival outcomes.
