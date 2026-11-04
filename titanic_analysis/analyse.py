import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load the titanic dataset from the data folder
df = pd.read_csv("data/Titanic-Dataset.csv")

# check the first few rows and basic info
print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values in dataset:")
print(df.isnull().sum())

# fill missing age values with the median age and drop unused columns
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.drop(columns=["Cabin", "Ticket", "Name"], errors="ignore")

# set the visual style for plots
sns.set_theme(style="whitegrid")

# plot 1: survival count based on gender
plt.figure(figsize=(7, 4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set2")
plt.title("Titanic Survival Analysis by Gender")
plt.xlabel("Gender")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# plot 2: survival count based on passenger class
plt.figure(figsize=(7, 4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set1")
plt.title("Titanic Survival Analysis by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# print percentage insights for your report
print("\nSurvival Percentage by Gender:")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival Percentage by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean() * 100)