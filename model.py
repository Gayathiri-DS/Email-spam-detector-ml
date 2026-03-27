import pandas as pd
import string
import pickle
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

nltk.download('stopwords')
df = pd.read_csv("Spam.csv")

columns = ['label', 'message']
df['label'] = df['label'].map({'ham':0, 'spam':1})

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['message'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred)*100)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred)*100)

pickle.dump(lr, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")

models = ['Logistic Regression' , 'Random Forest']
score = [accuracy_score(y_test,lr_pred),accuracy_score(y_test,rf_pred)]
plt.bar(models,score)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")

print("Graph saved as accuracy.png")
