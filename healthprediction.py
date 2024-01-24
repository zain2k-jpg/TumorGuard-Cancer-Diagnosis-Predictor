import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the Breast Cancer Wisconsin dataset 
df = pd.read_csv('data.csv')

# Separate features and target variable
X = df.drop('diagnosis', axis=1)  
y = df['diagnosis']
# Explore the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Separate features and target variable
X = df.drop('diagnosis', axis=1)  
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
