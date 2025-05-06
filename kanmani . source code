import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'fake_or_real_news.csv' with your dataset path)
# For demonstration purposes, let's create a small dummy dataset in memory
data = {'text': ["This is a real news article about the economy.",
                  "Breaking: Aliens land on Earth!",
                  "Scientists discover new treatment for cancer.",
                  "Shocking video shows politician taking bribe.",
                  "Local school wins national debate competition."],
        'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL']}
df = pd.DataFrame(data)

X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, and transform the testing data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize and train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(tfidf_train, y_train)

# Make predictions on the test set
y_pred = model.predict(tfidf_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example of predicting on new text
def predict_fake_news(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

new_text_1 = "Reports indicate a significant increase in unemployment rates."
prediction_1 = predict_fake_news(new_text_1)
print(f"\nPrediction for '{new_text_1}': {prediction_1}")

new_text_2 = "Secret government agency confirms existence of mermaids."
prediction_2 = predict_fake_news(new_text_2)
print(f"Prediction for '{new_text_2}': {prediction_2}")
