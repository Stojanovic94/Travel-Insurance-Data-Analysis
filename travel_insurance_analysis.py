import os
import pandas as pd
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Download and load data
path = kagglehub.dataset_download("mhdzahier/travel-insurance")
travel_csv = os.path.join(path, "travel insurance.csv")
df = pd.read_csv(travel_csv)
print("Loaded dataset from:", travel_csv)

# Step 2.1: Handle missing values
print("Missing values per column:")
print(df.isnull().sum())

df.dropna(inplace=True)
print("\nShape after dropping missing values:", df.shape)

# Step 2.2: Type conversion
df['Claim'] = df['Claim'].map({'Yes': 1, 'No': 0})
df['Age'] = df['Age'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Net Sales'] = df['Net Sales'].astype(float)
df['Commision (in value)'] = df['Commision (in value)'].astype(float)

print("\nData types after conversion:\n", df.dtypes)

# Step 2.3: One-hot encoding
categorical_cols = ['Agency', 'Agency Type', 'Distribution Channel',
                    'Product Name', 'Destination', 'Gender']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nShape after encoding:", df_encoded.shape)
print(df_encoded.head())

# Step 3.1: Claim status distribution
sns.countplot(data=df, x='Claim')
plt.title("Claim Status Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylabel("Number of Policies")
plt.show()

# Step 3.2: Sales by Agency Type
sns.countplot(data=df, x='Agency Type', order=df['Agency Type'].value_counts().index)
plt.title("Number of Sales by Agency Type")
plt.ylabel("Number of Policies")
plt.xlabel("Agency Type")
plt.show()

# Step 3.3: Age distribution by gender
sns.histplot(data=df, x='Age', hue='Gender', bins=30, kde=True, multiple='stack')
plt.title("Age Distribution by Gender")
plt.xlabel("Age")
plt.ylabel("Number of Clients")
plt.show()

# Step 3.4: Correlation heatmap
corr = df[['Duration', 'Net Sales', 'Commision (in value)', 'Age', 'Claim']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Variables")
plt.show()