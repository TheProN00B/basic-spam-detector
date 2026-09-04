import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #to convert text into numbers using tf-idf
from sklearn.naive_bayes import MultinomialNB #this is the model we will use to classify the messages
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv")

print(data.head())
print(data.shape)
print(data["label"].value_counts())


x = data["message"]  #these are the features/inputs
y = data["label"]   #these are the labels for those messages



x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, #this means that only 20% of the data will be used for testing and the rest will be used for training
    random_state=42
)   

vectorizer = TfidfVectorizer() #this will convert the text into numbers using tf-idf

X_train_vectorized = vectorizer.fit_transform(x_train)
X_test_vectorized = vectorizer.transform(x_test)

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("accuracy:", accuracy)



while True:
    message = input("\nEnter a message or 'exit' to quit; ")

    if message.lower() == "exit":
        break

    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    print("Prediction:", prediction[0])