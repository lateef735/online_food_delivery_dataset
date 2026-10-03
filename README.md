## Overview
Welcome to my analysis of online food ordering behavior and customer demographics. This project investigates the factors driving consumer preferences, ordering patterns, and satisfaction levels in the online food delivery market.

Using survey data detailing customer demographics, educational qualifications, family size, income brackets, and delivery feedback, this project explores key behavioral patterns. Through modular Jupyter Notebooks using Python, Pandas, Matplotlib, and Seaborn, I investigate demographic drivers of repeat ordering, feedback distribution across occupations, and the relationship between income and family size. i got this data from [kaggle.com/datasets/srisyra02/online-food-ordering-dataset](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset).

## The Questions
 Below are the primary analytical questions addressed in this project:

1. Who is ordering food online? What does the age and occupation distribution look like across the customer base?

2. How does customer feedback vary across occupations and genders? Are specific professional segments or genders driving negative reviews?

3. What is the relationship between income level, customer age, and family size? Do higher-earning households present different demographic profiles?

4. Does age play a role in repeat purchase decisions (Output)? Are younger demographics more likely to reorder compared to older cohorts?

5. How correlated are numeric household attributes? Is there a significant relationship between customer age and family size?

## Tools I Used
For this exploratory data analysis, I utilized the following tools:
Python: The core programming language used to manipulate data and generate statistical visualizations.
Pandas: Used for data ingestion, cleaning, reshaping, pivot table calculations, and crosstabs.
Matplotlib & Seaborn: Utilized together to build visual graphics, including count plots, KDE distribution histograms, box plots, and correlation heatmaps.
Jupyter Notebooks (via VS Code): Environment used to modularize the analysis into structured steps (1_data_clean_up.ipynb, charts.ipynb, age_and_family_size.ipynb, etc.).   
Git & GitHub: Utilized for version tracking and repository management.

## Data Preparation and Cleanup

This section details the preprocessing steps implemented to inspect data hygiene, handle columns, and structure categorical levels.

```# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset/online food delivery dataset.csv')

# Drop geospatial coordinates and unneeded columns
df_clean = df.drop(columns=['latitude', 'longitude', 'Pin code'])

# Check summary info and missing values
df_clean.info()
df_clean.isna().sum()
df_clean.duplicated().sum()
```
Dropping Non-Behavioral Columns: Coordinates (latitude, longitude) and postal data (Pin code) were removed to focus on customer socio-demographics and feedback.
Ordered Categoricals: Monthly income was structured as an ordered categorical series ('No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000') to ensure accurate progression across pivot tables and charts.

## The Analysis
## 1. Customer Age Distribution
Understanding the core age bracket of online food delivery users.

This is the amount of customers in different occupation filtered by the amount of positive or negative feedbacks they get 
```
sns.countplot(data=df_clean, x='Occupation', hue='Feedback')
plt.xticks(rotation=45); 
plt.legend()
plt.show()
```

![Occupation Count chart](images\Occupation_count_chart.png)

This is the age distribution between customers 
```
sns.histplot(df_clean['Age'], bins=15, kde=True)
```
![age_distribution_histogram](images\age_distribution_histogram.png)

## Insight
i. Age Distribution Analysis
* Age Distribution Analysis
* Peak Concentration: The mode is sharpest between 22 and 24 years, peaking around age 23 with over 70 individuals. Over 60% of the sample appears to fall within the 22–26 range.
* Tails & Boundary Artifacts: Participation drops off steadily after age 26. There is a small bump at the upper boundary (age 32–33), which frequently suggests either an open-ended survey grouping (e.g., "33 and above" lumped together) or slight tail distortion.
ii. Occupation Breakdown & Sentiment
* Dominant Demographic: Students constitute the vast majority of the dataset (over 200 total responses), followed by Employees (~120 total), Self Employeed (~55 total), and House wives (~10 total).
* Sentiment Ratio by Group:
  * Students: Overwhelmingly positive (~185 Positive vs. ~21 Negative), yielding a favorable response rate of nearly 90%.
  * Employees: Substantially positive (~85 Positive vs. ~33 Negative), with an approval rate around 72%.
  * Self Employed: Shows a similar pattern to employees (~38 Positive vs. ~16 Negative), roughly 70% positive.
  * House wives: Smallest sample size, but also heavily positive (~8 Positive vs. ~1 Negative)

Combined Synthesis
* Cohort Alignment: The heavy concentration of individuals aged 20–25 aligns directly with the predominance of the student demographic.
* Sentiment Skew: The overall positive bias in the outcome variable is largely driven by the student segment, which has both the highest volume and the highest proportion of positive ratings.

## 2. Feedback by Occupation

Investigating how satisfaction levels (Feedback) split across occupational groups (Occupation).

```
df_Feedback_percentage = df_clean.groupby('Occupation')['Feedback'].value_counts(normalize=True).unstack().round(2)

df_Feedback_plot = df_Feedback_percentage.plot(kind='barh')

plt.legend()
plt.show()
```

![feedback_by_occupation](images\feedback_by_occupation.png)

## Insight
* Normalized View vs. Raw Counts: Unlike the previous absolute count chart, this plot normalizes the responses into proportions (probabilities summing to 1.0 within each category), eliminating sample-size distortion and revealing the true sentiment rate per group.
* Two Distinct Sentiment Clusters:
    * Student: ~90% Positive vs. ~10% Negative.
    * House wife: ~89% Positive vs. ~11% Negative.

    Both groups show low friction and minimal dissatisfaction.
* Moderate-Satisfaction Cluster (~70–72% Positive):
    * Employee: ~72% Positive vs. ~28% Negative.
    * Self Employeed: ~70% Positive vs. ~30% Negative.
    Working professionals have roughly three times higher 
    negative feedback rates (~28–30%) compared to students and housewives (~10–11%).

Key Takeaway: Occupation is clearly correlated with sentiment. Individuals actively in the workforce (employees and business owners) express noticeably higher critical feedback, likely driven by tighter time constraints, higher service expectations, or differing price sensitivity compared to students and homemakers.

## 3. Feedback Proportion by Gender
Examining whether sentiment varies significantly by gender.

```
df_gender_feedback = (pd.crosstab(df_clean['Gender'], df_clean['Feedback'], normalize='index')
.mul(100)
.plot(kind='bar', stacked='True')
)

plt.show
```
![gender_feedback](images\gender_feedback.png)
## Insight
* Near-Identical Sentiment Profiles: Sentiment proportions show very little variation between genders
    * Female: ~84% Positive vs. ~16% Negative.
    * Male: ~80% Positive vs. ~20% Negative.
* Low Predictive Power: Compared to Occupation (where negative rates swung significantly from ~10% up to ~30%), Gender does not show strong separation for the target variable. In a classification or logistic regression model, gender will likely have a low feature importance/small coefficient unless it interacts with another variable (like occupation).    
## 4. Age vs. Repeat Ordering Behavior
Analyzing whether age impacts repeat ordering tendencies (Yes vs. No).
```
sns.boxplot(data=df_clean, x='Age', y='Output')
plt.show()
```
![age_output](images\age_output.png)
## Insight 
* Shift in Central Tendency:
    * The median age for "Yes" is 24 years (IQR: ~22 to 25).
    * The median age for "No" is 26 years (IQR: ~24 to 28).
    * Overall, respondents answering "No" skew noticeably older across quartiles.
* Spread and Outliers: 
    * "Yes" Group: Features a tighter interquartile range (IQR = ~3 years) with a few upper outliers extending from age 30 to 33 ($1.5 \times \text{IQR}$ beyond $Q_3$). The core concentration is strictly in the early 20s.  
    * "No" Group: Shows wider dispersion across the interquartile range (IQR = ~4 years) and whiskers spanning uniformly from age 19 up to 33, with no outliers detected.
* Synthesis with Prior Charts:
    * The lower median age in the "Yes" group mirrors the large student cohort (peaking at ages 22–24) who had ~90% positive sentiment.
    * Older respondents, who are more likely to be employed or self-employed, align directly with the higher prevalence of "No" / negative feedback.    
    * Age provides clear directional separation and will likely serve as a statistically significant feature in distinguishing between the two classes
## 5. Demographics Across Income Tiers
Evaluating how average customer age and family size change across monthly income brackets.
```
Income_order = [
    'No Income',
    'Below Rs.10000',
    '10001 to 25000',
    '25001 to 50000',
    'More than 50000'
]

df_clean['Monthly Income'] = pd.Categorical(
    df_clean['Monthly Income'],
    categories=Income_order,
    ordered=True
)

df_age_and_family_pivot = pd.pivot_table(df_clean, index='Monthly Income', values=['Age', 'Family size'], aggfunc='mean').round(1)

df_age_and_family_pivot.plot(kind='bar')

plt.ylabel('Age')
plt.show()

```
![age_and_family_size](images\age_and_family_size.png)
## Insight 
* Monotonic Relationship Between Age and Income:
    * Mean age increases progressively across every income tier:
        *No Income: ~23.1 years 
        * Below Rs. 10,000: ~23.8 years
        * 10,001 to 25,000: ~24.8 years 
        * 25,001 to 50,000: ~26.4 years
        * More than 50,000: ~27.3 years
    * This matches career lifecycle stages: respondents in the lowest/no-income brackets align with early-20s students, while higher earners skew into mid-to-late 20s as professional tenure increases.   
* Uniform Invariance in Family Size:
    * Average family size remains remarkably constant across all five income brackets, hovering flat between 3.0 and 3.5 members (with a tiny uptick to ~3.6 in the >50,000 bracket).
    * Income level does not systematically scale with household size in this sample.
* Feature Selection & Modeling Implications:
    * Multicollinearity: Age and Income have a direct positive association; including both in a regression model could introduce collinearity issues depending on the functional form.
    * Predictive Value: Family size shows low variance across income brackets, suggesting it will provide minimal discriminative power if used to segment income or feedback patterns compared to age and occupation. 
## 6. Correlation Analysis
```
sns.heatmap(df_clean.select_dtypes('number').corr(), annot=True, cmap='coolwarm')
``` 
![corrrelation_chart](images\corrrelation_chart.png)    
## Insight
* Weak Association ($r = 0.17$): Age and Family size share almost no linear relationship ($R^2 < 3\%$).  
* No Collinearity Risk: With $r = 0.17$, both variables can be safely used together in any model without multicollinearity concerns. 
* Independent Signals: Consistent with previous charts, family size remains flat regardless of age/income. 
## What I Learned

* Structured Categorical Sorting: Ordering ordinal string variables (such as income brackets) using pd.Categorical prevents sorting issues when producing pivot tables and grouped plots.
* Data Cleansing Practices: Removing redundant columns (Pin code, geospatial coordinates) early streamlines exploratory workflows and prevents noise in downstream analysis.
* Insight Synthesis: Combining count plots and normalized crosstabs clarifies customer proportions and sentiment rates across demographic categories.
## Challenges I Faced
* Data Hygiene and Redundancies: Unnamed index columns and extraneous location identifiers required careful verification against df.info() before charting.
* Class Imbalance: Students and positive feedback dominate the dataset, requiring normalized distributions and percentages rather than raw counts alone to avoid biased interpretations
## Conclusion
This exploratory project demonstrates that the primary audience for this food delivery platform is young adults, predominantly students and young professionals aged 22–26. While overall customer satisfaction remains high (~80%), employed professionals register a greater share of negative reviews, highlighting an opportunity to tailor premium delivery speeds and packaging quality to working professionals.