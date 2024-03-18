import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('fake_news.csv')
data.info()

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)

plt.imshow(confusion, cmap='Blues', interpolation='None')
plt.colorbar()
plt.xticks([0, 1], ['REAL', 'FAKE'])
plt.yticks([0, 1], ['REAL', 'FAKE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()