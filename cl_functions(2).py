#!/usr/bin/env python
# coding: utf-8

#### Libraries
import nltk
import numpy as np
import json
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the json file and store in a variable
with open('climate_change.json') as json_file:
    intents = json.load(json_file)

# Load the model and artifacts
model = load_model('cl_chatbotmodel.h5')
words = pickle.load(open('cl_words.pkl', 'rb'))
classes = pickle.load(open('cl_classes.pkl', 'rb'))

correctQues = ""

# Function to clean up the input sentence and lemmatize words
def clean_up_sentence(sentence):
    global correctQues
    correctQues = sentence
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to get BERT embeddings for a sentence using TensorFlow
def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return embeddings

# Predict the class using the trained model and BERT embeddings
def predict_class(sentence):
    embeddings = get_bert_embeddings(sentence)
    res = model.predict(embeddings)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get the response based on the predicted intent
def get_response(intents_list, intents_json):
    global correctQues
    result = None
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    ind = -1
    for index, i in enumerate(list_of_intents):
        if tag in i['tags']:
            result = i['responses']
            ind = index
            break
    return result, ind, correctQues

# Main function to handle user input and generate response
def main_(message: str):
    ints = predict_class(message)
    if len(ints) > 0:
        res = get_response(ints, intents)
        return res
    else:
        return ["I don't know about it", -1, ""]

# # Example usage
# message = "What are the impacts of climate change?"
# response = main_(message)
# print(response)
