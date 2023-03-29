import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load data
data = pd.read_csv('data.csv')

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)

# encode categorical features
cat_cols = ['cat_feature_1', 'cat_feature_2']
encoders = {}
for col in cat_cols:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])
    encoders[col] = encoder

# fit random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# evaluate performance
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Get feature importances
importances = rf.feature_importances_

# Print feature importances
for i, feature in enumerate(X.columns):
    print(f'{feature}: {importances[i]}')