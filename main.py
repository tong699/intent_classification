from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load CSV data
df = pd.read_csv('user_intent.csv')

# Train intent classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['user_message'])
clf = MultinomialNB().fit(X, df['user_intent'])

# FastAPI setup
app = FastAPI()

class UserInput(BaseModel):
    user_input: str

@app.post("/classify_intent")
def classify_intent(data: UserInput):
    user_text = data.user_input
    features = vectorizer.transform([user_text])
    intent = clf.predict(features)[0]
    return {"user_intent": intent}
