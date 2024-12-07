from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
from nltk.stem.porter import PorterStemmer
import pandas as pd
import google.generativeai as genai
import os
import string

app = FastAPI()

stemmer = PorterStemmer()

df = pd.read_csv("ques_ans.csv")
dataset = dict(zip(df["QUESTIONS"], df["ANSWERS"]))

questions = list(dataset.keys())
answers = list(dataset.values())

def stem(text):
    y = []
    for i in text.split():
        y.append(stemmer.stem(i))
    return " ".join(y)

stemmed_questions = [stem(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(stemmed_questions)

def correct_spelling(user_query):
    words = user_query.split()
    corrected_words = []
    for word in words:
        closest_match = process.extractOne(word, " ".join(questions).split(), scorer=fuzz.ratio)
        if closest_match[1] >= 80:
            corrected_words.append(closest_match[0])
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def postoffice_related(question):
    post_office_keywords = [
        "post office", "mail", "parcel", "stamp", "delivery", 
        "shipping", "courier", "postal", "package", "tracking",
        "address", "zip code", "pincode", "registered mail", 
        "speed post", "money order", "postcard", "letter", 
        "box", "mailbox", "sorting", "logistics", "dispatch", 
        "receiver", "sender", "postage", "envelope", "overseas", 
        "international mail", "domestic mail", "pickup", 
        "drop-off", "delivery status", "return to sender", 
        "signature required", "customs", "fragile", "priority mail"
    ]
    return any(keyword in question.lower() for keyword in post_office_keywords)

def fetch_gemini_answer(question):
    genai.configure(api_key=os.getenv("api_key"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Answer the following question: {question}")
    return response.text

def clean_query(user_query):
    user_query = user_query.strip().rstrip(string.punctuation)
    return user_query

def query_bot(user_query, threshold=0.5):
    cleaned_query = clean_query(user_query)
    corrected_query = correct_spelling(cleaned_query)
    stemmed_query = stem(corrected_query)
    user_query_vector = vectorizer.transform([stemmed_query])
    similarities = cosine_similarity(user_query_vector, question_vectors).flatten()
    max_similarity = similarities.max()
    best_match_idx = similarities.argmax()

    if max_similarity >= threshold:
        return {"answer": answers[best_match_idx]}
    else:
        if postoffice_related(cleaned_query):
            gemini_answer = fetch_gemini_answer(cleaned_query)
            dataset[cleaned_query] = gemini_answer
            updated_df = pd.DataFrame(dataset.items(), columns=["QUESTIONS", "ANSWERS"])
            updated_df.to_csv("ques_ans.csv", index=False)
            return {"answer": gemini_answer}
        else:
            raise HTTPException(status_code=404, detail="Sorry, I am not designed for it.")

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to the Query Bot, use endpoint /query"}

@app.post("/query")
def query(query: Query):
    response = query_bot(query.question)
    return response

