import streamlit as st, pandas as pd, numpy as np, matplotlib as mlp, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, plotly.graph_objs as go, plotly.express as ex, plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import chi2
import squarify
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import PCA

df= pd.read_csv('credit_risk.csv')

# load the first 5 rows of the dataset
#display column variables
st.write("Column Variables:", df.columns.tolist())

#Display the Dataframe
st.write("credit_risk.csv:", df)



# Title of the app
st.title('Statistical Analysis App')

# Display the dataset
st.subheader('Dataset:')
st.write(df)

# Descriptive statistics
st.subheader('Descriptive Statistics:')
st.write(df.describe())

# Correlation matrix (for numeric columns)
numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    st.subheader('Correlation Matrix:')
    st.write(df[numeric_cols].corr())

st.set_option('deprecation.showPyplotGlobalUse', False)

# Histograms
st.subheader('Histograms:')
for column in df.columns:
    if df[column].dtype == 'float64':
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        st.pyplot()

# Box plots
st.subheader('Box Plots:')
for column in df.columns:
    if df[column].dtype == 'float64':
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        st.pyplot()



# Check for missing values
missing_values = df.isnull().sum()

# Display missing values
st.write("Missing Values:")
st.write(missing_values)

# Display the DataFrame
st.write("credit_risk.csv:", df)

# Replace missing values with median
for col in df.columns:
    if df[col].dtype != object:  # Exclude non-numeric columns
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

# Display the DataFrame with missing values replaced
st.write("Data with Missing Values Replaced by Median:")
st.write(df)

# Function to check for duplicate values within each category
def check_duplicates_by_category(df):
    duplicate_values_by_category = {}
    for col in df.columns:
        if col != 'Category':
            continue
        unique_categories = df[col].unique()
        for category in unique_categories:
            subset_df = df[df[col] == category]
            duplicate_values = subset_df[subset_df.duplicated(subset=['Id'], keep=False)]
            if not duplicate_values.empty:
                if category not in duplicate_values_by_category:
                    duplicate_values_by_category[category] = duplicate_values
                else:
                    duplicate_values_by_category[category] = pd.concat([duplicate_values_by_category[category], duplicate_values])
    return duplicate_values_by_category

# Check for duplicate values within each category
duplicate_values_by_category = check_duplicates_by_category(df)

# Display duplicate values within each category
st.write("Duplicate Values within each Category:")
for category, duplicate_values in duplicate_values_by_category.items():
    st.write(f"Category: {category}")
    st.write(duplicate_values)

# Calculate debt-to-income (DTI) ratios
df['DTI'] = df['Amount'] / df['Income']

# Display the DataFrame with DTI ratios
st.write("Data with Debt-to-Income (DTI) Ratios:")
st.write(df)


# Define a threshold for DTI ratios
threshold = 0.3

# Create binary variable based on DTI threshold
df['HighDTI'] = (df['DTI'] > threshold).astype(int)

# Display the DataFrame with the binary variable
st.write("Data with Binary Variable (HighDTI):")
st.write(df)

# Correlation matrix (for numeric columns) with addition of DTI and HighDTI column
numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    st.subheader('Correlation Matrix:')
    st.write(df[numeric_cols].corr())



# Sample data
categories = [ 'Age', 'Home', 'Intent', 'Amount', 'Default']
values1 = np.random.randint(1, 10, size=len(categories))
values2 = np.random.randint(1, 10, size=len(categories))

# Create a bar chart
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(categories))

bar1 = ax.bar(index, values1, bar_width, label='Group 1')
bar2 = ax.bar(index + bar_width, values2, bar_width, label='Group 2')

# Add labels, title, and legend
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Dual Bar Chart')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

# Display the chart
st.pyplot(fig)

# Group data by loan status and purpose, and calculate total loan amount
grouped_data = df.groupby(['Status', 'Intent']).sum()

# Reset index to make 'Loan_Status' and 'Purpose' columns
grouped_data.reset_index(inplace=True)

# Create a treemap
plt.figure(figsize=(10, 8))
squarify.plot(sizes=grouped_data['Amount'], label=grouped_data.apply(lambda x: f"{x['Status']} - {x['Intent']}", axis=1), alpha=0.8)

# Add title and axis labels
plt.title('Loan Approval Status and Purpose Treemap')
plt.axis('off')

# Display the chart
st.pyplot(plt.gcf())

#Logistic regression model

# Prepare features (X) and target variable (y)
X = df.drop('Status', axis=1)  # Features
y = df['Status']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#Linear regressiopn model

# Prepare features (X) and target variable (y)
X = df.drop('Amount', axis=1)  # Features
y = df['Amount']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

#Random RandomForestClassifier
# Prepare features (X) and target variable (y)
X = df.drop('Status', axis=1)  # Features
y = df['Status']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Decision tree Model
# Prepare features (X) and target variable (y)
X = df.drop('Status', axis=1)  # Features
y = df['Status']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree (optional)
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=['Denied', 'Approved'], filled=True)
plt.show()

#Principal Component Analysis
# Prepare features (X) and target variable (y)
X = df.drop('Status', axis=1)  # Features
y = df['Status']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=10)  # Choose the number of principal components
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., Random Forest) using the PCA-transformed features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#line charts

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Age", "Income", "Amount"])

st.line_chart(chart_data)