from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Click here to win money",
    "Verify your bank account now",
    "Meeting scheduled for tomorrow",
    "Project report attached",
    "Congratulations! You won a prize",
    "Team meeting at 10 AM"
]

labels = [
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Phishing",
    "Safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
