import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("restaurant.csv")

print(df.head())

df.dropna(inplace=True) 
print(df.columns)
X = df['Cuisines'].astype(str)   
y = df['Cuisines'] 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
type(X_train)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Example text data
X = ["I love pizza", "Python is fun", "Machine learning"]
y = [1, 0, 1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = [str(x) for x in X_train]
X_test = [str(x) for x in X_test]

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

df = pd.read_csv('restaurant.csv')

df['Cuisines'] = df['Cuisines'].fillna('Unknown')

df['Cuisines'] = df['Cuisines'].str.lower()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Cuisines'])

X = df['Cuisines']

X = X.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)  # you can also try RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
unique_labels = np.unique(y_test)
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=label_encoder.classes_[unique_labels]))

misclassified = df.iloc[X_test.index][y_test != y_pred]
print("\nSample Misclassified Records:\n", misclassified.head())
