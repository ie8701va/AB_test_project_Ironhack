"""
Created on Sun Nov  3 17:54:36 2024

@author: ievav
"""
#%% Imports
# Import the libraries for data manipulation, visualization, statistical analysis, and machine learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import rcParams

# Set default color for plots using matplotlib's cycle property
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#2145b2'])

#%% Read and merge data

# Read data from CSV files into pandas 
experiment = pd.read_csv(r"data\experiment.csv")
geo = pd.read_csv(r"data\geo.csv")
purchases = pd.read_csv(r"data\purchases.csv")
users = pd.read_csv(r"data\users.csv")

# Merge the dataframes on the 'user_id' column
df = pd.merge(experiment, purchases, on="user_id", how="inner")
df = pd.merge(df, users, on="user_id", how="inner")

df.head(5)
#%% Overview the data

# Display the data types of each column 
df.dtypes

# Show summary statistics for all columns, including categorical ones
df.describe(include='all')

# (Commented cause my comp. is not powerful enough for this I guess)
# Calculate and display the total Gross Merchandise Value
# total_gmv = df['gmv'].sum()
# print(total_gmv)

#%% Check for duplicates

# Check for duplicates in the 'user_id' column
duplicates = df[df['user_id'].duplicated(keep=False)]  # keep=False shows all duplicates

# If duplicates are found, print them; otherwise, confirm no duplicates
if not duplicates.empty:
    print("Duplicates found in 'user_id':")
    print(duplicates)
else:
    print("No duplicates found in 'user_id'.")

#%% Map country

# Map country_id to country code using a predefined dictionary
country_mapping = {1: "FR", 2: "UK", 3: "IT"}

# Create a new 'country' column by mapping 'country_id' values using the dictionary from before
df["country"] = df["country_id"].map(country_mapping)

#%% Map gender

# Map gender values to numerical codes using a dictionary
gender_mapping = {"M": 1, "F": 2}

# Create a new 'gender_id' column, filling any missing values (NaN) with '3' (unknown)
df['gender_id'] = df['gender'].map(gender_mapping).fillna(3)

#%% Map test variant

# Map test variant (treatment/control) to numerical values
test_variant_mapping = {"treatment": 1, "control": 2}

# Create a new 'treatment_group' column by mapping the 'test_variant' column values
df['treatment_group'] = df['test_variant'].map(test_variant_mapping)

#%% Re-format time column

# Ensure 'timestamp' column is in datetime format for any subsequent time-related operations
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the hour from the timestamp and create a new 'hour' column
df['hour'] = df['timestamp'].dt.hour

#%% GMV and currency handling

# Clean GMV (Gross Merchandise Value) column: remove euro and pound signs
df['gmv'] = df['gmv'].str.replace('€', '', regex=False)  # Remove euro sign
df['gmv'] = df['gmv'].str.replace('£', '', regex=False)  # Remove pound sign

# Convert the cleaned 'gmv' column to numeric values (with coercion for any errors)
df['gmv'] = pd.to_numeric(df['gmv'], errors='coerce')

# Apply currency conversion (from GBP to EUR) for UK-based users
exchange_rate = 1.15  # Example exchange rate from GBP to EUR
df.loc[df['country'] == 'UK', 'gmv'] *= exchange_rate

#%% Gender Distribution

# Create a bar plot for the gender distribution of users
plt.figure(figsize=(10, 6))  # Set the figure size
gender_counts = df['gender'].value_counts(dropna=False)  # Count occurrences of each gender, including NaNs
sns.barplot(x=gender_counts.index.astype(str), y=gender_counts.values, color='#2145b2')  # Create bar plot

plt.title('Gender Distribution', fontsize=24, color='#2145b2')
plt.xlabel('Gender', fontsize=20, color='#2145b2')
plt.ylabel('Count', fontsize=20, color='#2145b2')
plt.xticks(fontsize=16, color='#2145b2')  # Customize tick font size
plt.yticks(fontsize=16, color='#2145b2')  # Customize tick font size

# Display the count values above the bars
for index, value in enumerate(gender_counts.values):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', color='#2145b2', fontsize=16)

# Remove plot spines (borders) for a cleaner look
sns.despine(left=True, bottom=True)

# Make the figure background transparent
#plt.gcf().patch.set_alpha(0)
#plt.gca().patch.set_alpha(0)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Majority are women. Fair amount of NANs (almost as many as men)

#%% Age Distribution

# Create a box plot for the age distribution of users
plt.figure(figsize=(10, 6))  # Set the figure size
sns.boxplot(data=df, x='age', color='#2145b2')  # Create box plot

# Customize the plot's appearance (title, labels, font size)
plt.title('Box Plot of Age Distribution', fontsize=24, color='#2145b2')
plt.xlabel('Age', fontsize=20, color='#2145b2')
plt.xticks(fontsize=22, color='#2145b2')
plt.yticks(fontsize=22, color='#2145b2')

# Remove plot spines (borders)
sns.despine(left=True, bottom=True)

# Make the figure background transparent
#plt.gcf().patch.set_alpha(0)
#plt.gca().patch.set_alpha(0)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Whiskers teenage years and past 70s. Majority are between 30-40y/o
#%% Country Distribution

# Create a bar plot for the country distribution of users
plt.figure(figsize=(10, 6))  # Set the figure size
country_counts = df['country'].value_counts()  # Count occurrences of each country
sns.barplot(x=country_counts.index.astype(str), y=country_counts.values, color='#2145b2')  # Create bar plot

# Customize the plot's appearance (title, labels, font size)
plt.title('Country Distribution', fontsize=24, color='#2145b2')
plt.xlabel('Country', fontsize=24, color='#2145b2')
plt.ylabel('Count', fontsize=24, color='#2145b2')
plt.xticks(fontsize=22, color='#2145b2')  # Customize tick font size
plt.yticks(fontsize=22, color='#2145b2')  # Customize tick font size

# Display the count values above the bars
for index, value in enumerate(country_counts.values):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', color='#2145b2', fontsize=22)

# Remove plot spines (borders)
sns.despine(left=True, bottom=True)

# Make the figure background transparent
#plt.gcf().patch.set_alpha(0)
#plt.gca().patch.set_alpha(0)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Majority are french

#%% Test variant distribution

# Create a bar plot for the test variant distribution
plt.figure(figsize=(10, 6))  # Set the figure size
test_variant_counts = df['test_variant'].value_counts()  # Count occurrences of each test variant
sns.barplot(x=test_variant_counts.index.astype(str), y=test_variant_counts.values, color='#2145b2')  # Create bar plot

# Customize the plot's appearance (title, labels, font size)
plt.title('Test Variant Distribution', fontsize=24, color='#2145b2')
plt.xlabel('Test Variant', fontsize=24, color='#2145b2')
plt.ylabel('Count', fontsize=24, color='#2145b2')
plt.xticks(fontsize=22, color='#2145b2')  # Customize tick font size
plt.yticks(fontsize=22, color='#2145b2')  # Customize tick font size

# Display the count values above the bars
for index, value in enumerate(test_variant_counts.values):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', color='#2145b2', fontsize=22)

# Remove plot spines (borders)
sns.despine(left=True, bottom=True)

# Make the figure background transparent
#plt.gcf().patch.set_alpha(0)
#plt.gca().patch.set_alpha(0)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Test variant numbers are equally distributed accross treatmnet and control groups.

#%% Hour of the day sales distribution

# Create a bar plot for the hour distribution of user activity
plt.figure(figsize=(10, 6))  # Set the figure size
hour_counts = df['hour'].value_counts().sort_index()  # Count occurrences of each hour
sns.barplot(x=hour_counts.index, y=hour_counts.values, color='#2145b2')  # Create bar plot

# Customize the plot's appearance (title, labels, font size)
plt.title('Hour Distribution of User Activity', fontsize=24, color='#2145b2')
plt.xlabel('Hour', fontsize=24, color='#2145b2')
plt.ylabel('Count', fontsize=24, color='#2145b2')
plt.xticks(fontsize=22, color='#2145b2')  # Customize tick font size
plt.yticks(fontsize=22, color='#2145b2')  # Customize tick font size

# # Display the count values above the bars
# for index, value in enumerate(hour_counts.values):
#     plt.text(index, value + 0.5, str(value), ha='center', va='bottom', color='#2145b2', fontsize=22)

# Remove plot spines (borders)
sns.despine(left=True, bottom=True)

# Make the figure background transparent
#plt.gcf().patch.set_alpha(0)
#plt.gca().patch.set_alpha(0)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# As expected, the sales are really low at nigh time hours with peaks right in the moring 9-10am and afterwork hours 5-9pm. 


#%% Exploratory analysis - correlation heatmap

# Drop the 'treatment' column
correlation_matrix = df.drop(columns=['test_variant', 'gender', 'country']).corr()

# Display the correlation matrix
correlation_matrix

# Specify the columns you want to select
columns_to_select = ['gmv', 'country_id', 'age', 'hour', 'gender_id', 'treatment_group']

# Create a new DataFrame with the selected columns
new_df = df[columns_to_select]

# Calculate correlation matrix
corr = new_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()

# Hour of the day nad gmv seems to have the highest correlation. Country ID also could be something to look into. 
# We already know that as the day progresses there are different buyer behaviors. 
# SOOO in the following analysis, I will look into the changes over time across different countries. To see if we can forecast chance in different treatment variant groups. 

#%% GMV focused analyses. Plotting GMV against AGE

# Step 1: Create Age Bins
# For float age, we'll round down to the nearest whole number to create 1-year bins.
df['age_bins'] = pd.cut(df['age'], bins=range(0, int(df['age'].max()) + 2), right=False)

# Step 2: Group By Age and Calculate Average GMV
average_gmv_by_age = df.groupby('age_bins')['gmv'].mean().reset_index()

# Step 3 Convert the age bins to a more readable format for plotting
average_gmv_by_age['age'] = average_gmv_by_age['age_bins'].apply(lambda x: x.left)

# Step 4: Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='gmv', data=average_gmv_by_age, color='teal')
plt.title('Average GMV vs Age')
plt.xlabel('Age')
plt.ylabel('Average GMV')
plt.tight_layout()
plt.show()

# Older clients tend to generate more revenue.

#%% GMV focused analyses. Plotting cumulative GMV against TIME

# Step 1: Group by hour and sum GMV
cumulative_gmv = df.groupby('hour')['gmv'].sum().reset_index()

# Step 2: Calculate the cumulative sum
cumulative_gmv['cumulative_gmv_hours'] = cumulative_gmv['gmv'].cumsum()

# Step 3: Plot the cumulative GMV
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='cumulative_gmv_hours', data=cumulative_gmv, color='teal', marker='o')
plt.title('Cumulative GMV Throughout the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Cumulative GMV')
plt.xticks(range(0, 24))  # Set x-ticks for each hour
plt.grid()  # Optional: Add grid for better readability
plt.tight_layout()
plt.show()  # Display the plot

#%% GMV focused analyses. Plotting cumulative GMV against COUNTRY

# Since I saw the country had a -0.12 correlation with gmv, I will plot this too. 

# Step 1: Group by hour and country, then sum GMV
cumulative_gmv_by_country = df.groupby(['hour', 'country'])['gmv'].sum().reset_index()

# Step 2: Calculate the cumulative sum for each country
cumulative_gmv_by_country['cumulative_gmv_hours'] = cumulative_gmv_by_country.groupby('country')['gmv'].cumsum()

# Step 3: Plot the cumulative GMV with separate lines for each country
plt.figure(figsize=(10, 6))
sns.lineplot(
    x='hour', 
    y='cumulative_gmv_hours', 
    hue='country', 
    data=cumulative_gmv_by_country, 
    marker='o'
)
plt.title('Cumulative GMV Throughout the Day by Country')
plt.xlabel('Hour of the Day')
plt.ylabel('Cumulative GMV')
plt.xticks(range(0, 24))  # Set x-ticks for each hour
plt.grid()  # Optional: Add grid for better readability
plt.legend(title='Country')  # Add legend for countries
plt.tight_layout()
plt.show()  # Display the plot

# It is visible that France is generatinf more GMV compared to Italy and UK. 
# The pattern of activation throughout the day is the same though. 
# As in there is slow growth during night, and peaks before and after working hours 
# (with an exception fro between 6-7pm --> Commuting?).. 

#%% AB test analysis - overall

# Calculate GMV statistics for each group
gmv_summary = df.groupby('treatment_group')['gmv'].agg(['mean', 'median', 'sum', 'count']).reset_index()
print(gmv_summary)

# Violin Plot of GMV by Treatment Group
plt.figure(figsize=(10, 6))
sns.violinplot(x='treatment_group', y='gmv', data=df,  color='#2145b2')  # Set custom colors for treatment groups
plt.ylabel('GMV', fontsize=24, color='#2145b2')  
plt.xticks(fontsize=22, color='#2145b2', ticks=[0, 1], labels=['Experimental', 'Control'])  # X-axis ticks font size and color
plt.yticks(fontsize=22, color='#2145b2')  
plt.text(1, 1.05, 't = 113.51, p < .001', ha='right', fontsize=22, color='#2145b2', transform=plt.gca().transAxes)

sns.despine(left=True, bottom=True)  

#plt.gcf().patch.set_alpha(0)  # Figure background transparency
#plt.gca().patch.set_alpha(0)  # Plot area transparency

plt.tight_layout()  
plt.show()


# Separate data by treatment groups
experimental_group = df[df['treatment_group'] == 1]
control_group = df[df['treatment_group'] == 2]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(experimental_group['gmv'].dropna(), control_group['gmv'].dropna())
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Interpret the results
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
    
# Reject the null hypothesis: There is a significant difference between the test variant groups.
        
#%% AB test analysis - across different countries

# Group by treatment_group and country, calculating mean, median, and count of GMV
country_treatment_gmv_stats = df.groupby(['treatment_group', 'country'])['gmv'].agg(['mean', 'median', 'count']).reset_index()
print(country_treatment_gmv_stats)

plt.figure(figsize=(14, 8))

# Custom color palette for treatment groups
custom_palette = { 1: '#2145b2', 2: '#e0ca27'}   # 0 = Control, 1 = Experimental (or adjust according to your data)

# Create violin plot 
sns.violinplot(data=df, x='country', y='gmv', hue='treatment_group', split=True, inner=None, palette=custom_palette)

# Overlay summary statistics (mean) with dots, using the same color for the mean points as the group color
sns.pointplot(data=df, x='country', y='gmv', hue='treatment_group', estimator='mean', dodge=0.5, markers="o", join=False, palette = custom_palette, ci=None, legend=False)

# Set title and axis labels with customized font size and color
plt.title('GMV Distribution by Country and Treatment Group', fontsize=24, color='#2145b2')
plt.xlabel('Country', fontsize=24, color='#2145b2')
plt.ylabel('GMV', fontsize=24, color='#2145b2')

plt.xticks(fontsize=22, color='#2145b2')
plt.yticks(fontsize=22, color='#2145b2')

# Adjust legend appearance (only for violin plot groups)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title='Treatment Group', loc='upper right', fontsize=20, title_fontsize=22)
plt.setp(plt.gca().get_legend().get_texts(), color='#2145b2')  # Set legend text color
plt.setp(plt.gca().get_legend().get_title(), color='#2145b2')  # Set legend title color

sns.despine(left=True, bottom=True)
#plt.gcf().patch.set_alpha(0)  # Make figure background transparent
#plt.gca().patch.set_alpha(0)  # Make plot area transparent

plt.tight_layout()
plt.show()


# Get unique countries
countries = df['country'].unique()

# Loop through each country and perform ANOVA
for country in countries:
    print(f"ANOVA for {country}:")
    gmv_by_treatment = [df[(df['country'] == country) & (df['treatment_group'] == group)]['gmv'] for group in df['treatment_group'].unique()]
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*gmv_by_treatment)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Significant difference in GMV between treatment groups in this country.")
    else:
        print("No significant difference in GMV between treatment groups in this country.")
    print()
    
# Italy and France experimental groups have generated significantly higher GMV compared to control groups within these countries. 
    
#%% AB test analysis - across different age bins
# Group by age_bins and treatment_group, calculating the mean GMV
average_gmv_by_age_treatment = df.groupby(['age_bins', 'treatment_group'])['gmv'].mean().reset_index()


plt.figure(figsize=(14, 8))
sns.barplot(data=average_gmv_by_age_treatment, x='age_bins', y='gmv', hue='treatment_group', palette='Set2')
plt.title('Average GMV by Age Bins and Treatment Group')
plt.xlabel('Age Bins')
plt.ylabel('Average GMV')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.legend(title='Treatment Group')

current_ticks = plt.xticks()[0]  # Get the tick positions
plt.xticks(ticks=current_ticks[::3])  # Keep only every third tick
plt.xlim(left=13)  # Set the left limit of the x-axis to 13
plt.tight_layout()
plt.show()

# Get unique age bins
age_bins_unique = df['age_bins'].unique()

# Loop through each age bin and perform ANOVA
for age_bin in age_bins_unique:
    print(f"ANOVA for Age Bin: {age_bin}")
    gmv_by_treatment = [
        df[(df['age_bins'] == age_bin) & (df['treatment_group'] == group)]['gmv']
        for group in df['treatment_group'].unique()
    ]
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*gmv_by_treatment)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Significant difference in GMV between treatment groups for this age bin.")
    else:
        print("No significant difference in GMV between treatment groups for this age bin.")
    print()

# Based on the output, the general trend is that there are significant test variant group differences in until approx. 70 y/o. 

#%% Preparing for a regression - non-spenders across different countries

# Filter data for users with gmv == 0
gmv_zero_df = df[df['gmv'] == 0]

# Count the number of users with gmv == 0 by country
country_counts_non_payers = gmv_zero_df['country'].value_counts()

# Count the total number of customers per country
total_per_country_customers = df.groupby('country').size()

# Calculate the non-payer ratio per country
non_payer_ratio_per_country = country_counts / total_per_country_customers

# Display the result for better readability
non_payer_ratio_df = non_payer_ratio_per_country.reset_index()
non_payer_ratio_df.columns = ['country', 'non_payer_ratio']

print(non_payer_ratio_df)


# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Users with GMV = 0 by Country')
plt.show()

# In every country there are proportionally as many non-spenders as there are overall buyers.
# So it will not make a difference if we remove these people from the regression that follows. 
    
#%% Linear regression predicting GMV (age as predictor)

# Filter data for users with GMV not equal to 0
gmv_without_zero_df = df[df['gmv'] != 0]

# Scatterplot representing a general trend for age in different countries
plt.figure(figsize=(10, 6))
sns.scatterplot(data=gmv_without_zero_df, x='age', y='gmv', hue='country', palette='viridis', alpha=0.7)

# Initialize a dictionary to store the regression results
regression_results = {}

# First, fit the regressions and store results
for country in gmv_without_zero_df['country'].unique():
    for variant in ['treatment', 'control']:
        # Filter data for the current country and test_variant
        country_variant_data = gmv_without_zero_df[(gmv_without_zero_df['country'] == country) & 
                                                   (gmv_without_zero_df['test_variant'] == variant)]
        
        if country_variant_data.empty:
            continue  # Skip if no data for this country and variant
        
        X = country_variant_data[['age']].values
        y = country_variant_data['gmv'].values

        # Fit the simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Generate x values for plotting the regression line
        age_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        
        # Predict GMV values over the age range
        gmv_pred = model.predict(age_range)

        # Calculate R² and MSE
        r2_score_val = model.score(X, y)  # R² value
        mse_val = mean_squared_error(y, model.predict(X))  # MSE value
        
        # Store the results in the dictionary
        regression_results[(country, variant)] = {
            'age_range': age_range,
            'gmv_pred': gmv_pred,
            'r2': r2_score_val,
            'mse': mse_val
        }

# There is a clear trend with France generating most revenue, followed by Italy and then UK

# Now, plot the results
# Create two subplots (side by side) for 'treatment' and 'control'
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Set titles for the subplots
axes[0].set_title('Experimental Group')
axes[1].set_title('Control Group')

# Loop through each country and plot the regression for 'treatment' and 'control' on separate subplots
for country in gmv_without_zero_df['country'].unique():
    for variant, ax in zip(['treatment', 'control'], axes):
        # Get the regression results from the dictionary
        key = (country, variant)
        if key not in regression_results:
            continue
        
        # Extract the regression results
        results = regression_results[key]
        age_range = results['age_range']
        gmv_pred = results['gmv_pred']
        r2 = results['r2']
        mse = results['mse']
        
        # Set line style based on test_variant
        linestyle = '-' if variant == 'treatment' else '--'
        
        # Set color for each country based on the hue palette
        color = sns.color_palette('viridis', len(gmv_without_zero_df['country'].unique()))[list(gmv_without_zero_df['country'].unique()).index(country)]
        
        # Plot the simple linear regression line with the specified line style and color
        ax.plot(age_range, gmv_pred, linestyle=linestyle, label=f'{country}', linewidth=2, color=color)

# Customize plot labels and titles
axes[0].set_xlabel('Age')
axes[0].set_ylabel('GMV')
axes[1].set_xlabel('Age')

# Position the legend outside the plot area and ensure it's grouped by country
axes[0].legend(title='Country ID', loc='upper left', bbox_to_anchor=(1.05, 1))
axes[1].legend(title='Country ID', loc='upper left', bbox_to_anchor=(1.05, 1))

# Display the plot
plt.tight_layout()
plt.show()

# In the experimental group France is clearly different. French elderly are more affected by the implementation of the feature.
# But the main effect of age is there - the older the guyer, the more GMV it will generate

#%% Time Series Analysis - Forecasting GMV growth with ARIMA in general

# I wanted to prefict the next 24 hour growth in GMV

# Calculate hourly GMV by taking the difference of the cumulative GMV
cumulative_gmv['hourly_gmv'] = cumulative_gmv['cumulative_gmv_hours'].diff().fillna(cumulative_gmv['cumulative_gmv_hours'].iloc[0])

# Fit an ARIMA model to the hourly GMV data
model = ARIMA(cumulative_gmv['hourly_gmv'], order=(2, 2, 1))
model_fit = model.fit()

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24)

# Create a DataFrame for the forecasted hours
forecast_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv': forecast})

# Calculate cumulative GMV based on forecasted hourly GMV
forecast_hours['cumulative_forecast_gmv'] = forecast_hours['hourly_gmv'].cumsum() + cumulative_gmv['cumulative_gmv_hours'].iloc[-1]

# Combine actual and forecasted data for plotting
combined_gmv = pd.concat([cumulative_gmv[['hour', 'cumulative_gmv_hours']],
                          forecast_hours[['hour', 'cumulative_forecast_gmv']].rename(columns={'cumulative_forecast_gmv': 'cumulative_gmv'})])

# Plot the actual and forecasted cumulative GMV
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='cumulative_gmv_hours', data=combined_gmv, color='teal', marker='o', label='Actual Cumulative GMV')
sns.lineplot(x='hour', y='cumulative_forecast_gmv', data=forecast_hours, color='orange', marker='o', label='Forecasted Cumulative GMV')
plt.title('Actual and Forecasted Cumulative GMV')
plt.xlabel('Hour of the Day')
plt.ylabel('Cumulative GMV')
plt.xticks(range(0, 48, 2))  # Set x-ticks for each 2 hours
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Although with ARIMA I could only forecase linear growth, the pattern is there. 
#%%Time Series Analysis - Forecasting GMV growth with ARIMA for the AB groups

# Now let's see if there's a different in this groth across different treatment groups. 

cumulative_gmv_AB = df.groupby(['test_variant', 'hour'])['gmv'].sum().reset_index()

# Separate into two DataFrames based on 'test_variant'
cumulative_gmv_A = cumulative_gmv_AB[cumulative_gmv_AB['test_variant'] == 'control']
cumulative_gmv_B = cumulative_gmv_AB[cumulative_gmv_AB['test_variant'] == 'treatment']

# Step 2: Calculate the cumulative sum
cumulative_gmv_A['cumulative_gmv_A'] = cumulative_gmv_A['gmv'].cumsum()
cumulative_gmv_B['cumulative_gmv_B'] = cumulative_gmv_B['gmv'].cumsum()

# Calculate hourly GMV by taking the difference of the cumulative GMV
cumulative_gmv_A['hourly_gmv_A'] = cumulative_gmv_A['cumulative_gmv_A'].diff().fillna(cumulative_gmv_A['cumulative_gmv_A'].iloc[0])
cumulative_gmv_B['hourly_gmv_B'] = cumulative_gmv_B['cumulative_gmv_B'].diff().fillna(cumulative_gmv_B['cumulative_gmv_B'].iloc[0])

# Fit an ARIMA model to the hourly GMV data
model_A = ARIMA(cumulative_gmv_A['hourly_gmv_A'], order=(2, 2, 1))  # Adjust (p, d, q) as needed
model_B = ARIMA(cumulative_gmv_B['hourly_gmv_B'], order=(2, 2, 1))  # Adjust (p, d, q) as needed


model_A_fit = model_A.fit()
model_B_fit = model_B.fit()

# Forecast the next 24 hours
forecast_A = model_A_fit.forecast(steps=24)
forecast_B = model_B_fit.forecast(steps=24)


# Create a DataFrame for the forecasted hours
forecast_A_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv_A': forecast_A})
forecast_B_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv_B': forecast_B})


# Calculate cumulative GMV based on forecasted hourly GMV
forecast_A_hours['cumulative_A_forecast_gmv'] = forecast_A_hours['hourly_gmv_A'].cumsum() + cumulative_gmv_A['cumulative_gmv_A'].iloc[-1]
forecast_B_hours['cumulative_B_forecast_gmv'] = forecast_B_hours['hourly_gmv_B'].cumsum() + cumulative_gmv_B['cumulative_gmv_B'].iloc[-1]

# Custom color palette for control and experimental groups
custom_palette = {'control': '#2145b2', 'treatment': '#e0ca27'}

# Plotting both control (A) and treatment (B) on the same plot with custom colors
plt.figure(figsize=(12, 6))

sns.lineplot(x='hour', y='cumulative_gmv_A', data=cumulative_gmv_A, color=custom_palette['control'], marker='o', label='Actual Control', linewidth=2, alpha=0.35)
sns.lineplot(x='hour', y='cumulative_gmv_B', data=cumulative_gmv_B, color=custom_palette['treatment'], marker='o', label='Actual Experimental', linewidth=2, alpha=0.35)
sns.lineplot(x='hour', y='cumulative_A_forecast_gmv', data=forecast_A_hours, color=custom_palette['control'], marker='o', label='Forecasted Control', linewidth=2, linestyle='-', alpha=0.7)
sns.lineplot(x='hour', y='cumulative_B_forecast_gmv', data=forecast_B_hours, color=custom_palette['treatment'], marker='o', label='Forecasted Experimental', linewidth=2, linestyle='-', alpha=0.7)

# Customize the title and axis labels
plt.title('Actual and Forecasted Cumulative GMV for Control and Treatment Groups', fontsize=24, color='#2145b2')
plt.xlabel('Hours since feature implementation', fontsize=20, color='#2145b2')
plt.ylabel('Cumulative GMV', fontsize=20, color='#2145b2')

plt.xticks(range(0, 48, 2), fontsize=18, color='#2145b2')  # X-ticks every 2 hours with font size 18
plt.yticks(fontsize=18, color='#2145b2')  # Y-ticks font size and color
plt.grid(True, alpha=0.3)  # Light grid lines

plt.legend(fontsize=16, title_fontsize=18, loc='upper left', frameon=False)

sns.despine(left=True, bottom=True)
#plt.gcf().patch.set_alpha(0)  # Figure background transparency
#plt.gca().patch.set_alpha(0)  # Plot area transparency

plt.tight_layout()
plt.show()

# The control and treatment cumulative GMV seems to start to diverge but slowly. 

#%%Time Series Analysis - Forecasting GMV growth with ARIMA in experimental group across countries

# We saw a clear country-wise difference in revenue growth.
# Let's see if there is truly difference in how GMV accumulates in experimental group but across countries.

df_FR = df[df['country'] == 'FR']
df_UK = df[df['country'] == 'UK']
df_IT = df[df['country'] == 'IT']


FR_cumulative_gmv_AB = df_FR.groupby(['test_variant', 'hour'])['gmv'].sum().reset_index()
UK_cumulative_gmv_AB = df_UK.groupby(['test_variant', 'hour'])['gmv'].sum().reset_index()
IT_cumulative_gmv_AB = df_IT.groupby(['test_variant', 'hour'])['gmv'].sum().reset_index()


cumulative_gmv_B_FR = FR_cumulative_gmv_AB[FR_cumulative_gmv_AB['test_variant'] == 'treatment']
cumulative_gmv_B_UK = UK_cumulative_gmv_AB[UK_cumulative_gmv_AB['test_variant'] == 'treatment']
cumulative_gmv_B_IT = IT_cumulative_gmv_AB[IT_cumulative_gmv_AB['test_variant'] == 'treatment']


cumulative_gmv_B_FR['cumulative_gmv_B_FR'] = cumulative_gmv_B_FR['gmv'].cumsum()
cumulative_gmv_B_UK['cumulative_gmv_B_UK'] = cumulative_gmv_B_UK['gmv'].cumsum()
cumulative_gmv_B_IT['cumulative_gmv_B_IT'] = cumulative_gmv_B_IT['gmv'].cumsum()


cumulative_gmv_B_FR['hourly_gmv_B_FR'] = cumulative_gmv_B_FR['cumulative_gmv_B_FR'].diff().fillna(cumulative_gmv_B_FR['cumulative_gmv_B_FR'].iloc[0])
cumulative_gmv_B_UK['hourly_gmv_B_UK'] = cumulative_gmv_B_UK['cumulative_gmv_B_UK'].diff().fillna(cumulative_gmv_B_UK['cumulative_gmv_B_UK'].iloc[0])
cumulative_gmv_B_IT['hourly_gmv_B_IT'] = cumulative_gmv_B_IT['cumulative_gmv_B_IT'].diff().fillna(cumulative_gmv_B_IT['cumulative_gmv_B_IT'].iloc[0])



model_B_FR = ARIMA(cumulative_gmv_B_FR['hourly_gmv_B_FR'], order=(2, 2, 1))  # Adjust (p, d, q) as needed
model_B_UK = ARIMA(cumulative_gmv_B_UK['hourly_gmv_B_UK'], order=(2, 2, 1))  # Adjust (p, d, q) as needed
model_B_IT = ARIMA(cumulative_gmv_B_IT['hourly_gmv_B_IT'], order=(2, 2, 1))  # Adjust (p, d, q) as needed

model_B_FR_fit = model_B_FR.fit()
model_B_UK_fit = model_B_UK.fit()
model_B_IT_fit = model_B_IT.fit()


forecast_B_FR = model_B_FR_fit.forecast(steps=24)
forecast_B_UK = model_B_UK_fit.forecast(steps=24)
forecast_B_IT = model_B_IT_fit.forecast(steps=24)


forecast_B_FR_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv_B_FR': forecast_B_FR})
forecast_B_UK_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv_B_UK': forecast_B_UK})
forecast_B_IT_hours = pd.DataFrame({'hour': range(24, 48), 'hourly_gmv_B_IT': forecast_B_IT})


forecast_B_FR_hours['cumulative_B_FR_forecast_gmv'] = forecast_B_FR_hours['hourly_gmv_B_FR'].cumsum() + cumulative_gmv_B_FR['cumulative_gmv_B_FR'].iloc[-1]
forecast_B_UK_hours['cumulative_B_UK_forecast_gmv'] = forecast_B_UK_hours['hourly_gmv_B_UK'].cumsum() + cumulative_gmv_B_UK['cumulative_gmv_B_UK'].iloc[-1]
forecast_B_IT_hours['cumulative_B_IT_forecast_gmv'] = forecast_B_IT_hours['hourly_gmv_B_IT'].cumsum() + cumulative_gmv_B_IT['cumulative_gmv_B_IT'].iloc[-1]


# Plotting both control (A) and treatment (B) on the same plot with custom colors
plt.figure(figsize=(12, 6))

sns.lineplot(x='hour', y='cumulative_gmv_B_FR', data=cumulative_gmv_B_FR, color='green', marker='o', linewidth=2, alpha=0.35)
sns.lineplot(x='hour', y='cumulative_gmv_B_UK', data=cumulative_gmv_B_UK,  color='red', marker='o', linewidth=2, alpha=0.35)
sns.lineplot(x='hour', y='cumulative_gmv_B_IT', data=cumulative_gmv_B_IT, color='blue', marker='o', linewidth=2, alpha=0.35)

sns.lineplot(x='hour', y='cumulative_B_FR_forecast_gmv', data=forecast_B_FR_hours, color='green', marker='o', label='Experimental France', linewidth=2, linestyle='-', alpha=0.7)
sns.lineplot(x='hour', y='cumulative_B_UK_forecast_gmv', data=forecast_B_UK_hours, color='red', marker='o', label='Experimental United Kingdom', linewidth=2, linestyle='-', alpha=0.7)
sns.lineplot(x='hour', y='cumulative_B_IT_forecast_gmv', data=forecast_B_IT_hours, color='blue', marker='o', label='Experimental Italy', linewidth=2, linestyle='-', alpha=0.7)

# Customize the title and axis labels
#plt.title('Actual and Forecasted Cumulative GMV for Different Countries', fontsize=24, color='#2145b2')
plt.xlabel('Hours since feature implementation', fontsize=20, color='#2145b2')
plt.ylabel('Cumulative GMV', fontsize=20, color='#2145b2')

plt.xticks(range(0, 48, 2), fontsize=18, color='#2145b2')  # X-ticks every 2 hours with font size 18
plt.yticks(fontsize=18, color='#2145b2')  # Y-ticks font size and color
plt.grid(True, alpha=0.3)  # Light grid lines

plt.legend(fontsize=16, title_fontsize=18, loc='upper left', frameon=False)

sns.despine(left=True, bottom=True)
#plt.gcf().patch.set_alpha(0)  # Figure background transparency
#plt.gca().patch.set_alpha(0)  # Plot area transparency

plt.tight_layout()
plt.show()

# As expected, not only that experimental group is forecasted to gro revenues faster than control
# But also, in France it is expected to happen much faster. 
# Limitation of this analysis - we only have 24 hours worth of data, so the predictions do not apply over long periods of time 
 
#%% CONCLUSION
# Introduction of a feature overall was success
# Users in France are more responsive to personalized deals, possibly due to stronger engagement with localized promotions or a higher willingness to explore deals.
# While we should expand the feature in similar markets, countries with weaker responses might need further investigation  (e.g., market maturity, cultural differences,  user engagement levels) that could influence the feature's effectiveness.