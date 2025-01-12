# AB_test_project_Ironhack
A/B Test Analysis Project for Ironhack

# AB Test Analysis and Regression Modeling on GMV Data

This script performs a detailed analysis of GMV (Gross Merchandise Value) data across various treatment groups, countries, and age bins. The analysis involves multiple steps, including statistical testing (t-tests, ANOVA), data visualization (violin plots, bar plots, scatterplots), and regression modeling. The ultimate goal is to understand the differences in GMV across various factors such as treatment group, country, and age, and to perform a forecast using ARIMA for future GMV growth.

## Steps Covered in the Script:

### 1. **AB Test Analysis - Overall**
   - The script first calculates the GMV statistics for each treatment group (Experimental vs Control) and displays summary statistics (mean, median, sum, count).
   - A violin plot is then created to visually represent the distribution of GMV values across the two groups.
   - An independent t-test is performed to assess whether there is a significant difference in GMV between the experimental and control groups.

### 2. **AB Test Analysis - Across Different Countries**
   - GMV statistics (mean, median, count) are calculated for each treatment group across different countries.
   - A violin plot and point plot are created to visualize the GMV distribution by country and treatment group, with a customized color palette.
   - ANOVA tests are performed to determine whether there is a significant difference in GMV between treatment groups within each country.

### 3. **AB Test Analysis - Across Different Age Bins**
   - The GMV is averaged by age bin and treatment group.
   - A bar plot is created to visualize the average GMV by age bins and treatment group.
   - ANOVA tests are performed for each age bin to check for significant differences between the treatment groups.

### 4. **Preparing for Regression Analysis - Non-Spenders Across Different Countries**
   - The script filters users with a GMV of 0 (non-spenders) and calculates the non-payer ratio per country.
   - A pie chart is created to display the distribution of users with GMV = 0 by country.

### 5. **Linear Regression - Predicting GMV (Age as a Predictor)**
   - A linear regression model is fit to predict GMV based on age, excluding non-spenders (GMV = 0).
   - Scatterplots and regression lines are plotted for each country and treatment group.
   - The regression results (R², MSE) are calculated and stored for later analysis.

### 6. **Time Series Analysis - Forecasting GMV Growth with ARIMA**
   - The script prepares for time series forecasting using ARIMA to predict GMV growth for the next 24 hours. (This section is currently a placeholder for future work.)

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scipy
- Statsmodels

## How to Run the Script

1. Clone or download the repository to your local machine.
2. Ensure all required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy statsmodels
