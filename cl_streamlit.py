import nltk
import numpy as np
import json
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
from transformers.models.bert.modeling_tf_bert import TFBertModel
import streamlit as st
from streamlit_chat import message

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


# Streamlit app
st.title("Climate Change Info Chat-Bot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'index' not in st.session_state:
    st.session_state.index = []

if 'ind' not in st.session_state:
    st.session_state.ind = -1

if 'question' not in st.session_state:
    st.session_state.question = ""

def get_text():
    """
    Get user input in a text box and store in input_text variable
    """
    input_text = st.text_input("Question: ", "", key="input")
    return input_text

# Store user input
user_input = get_text()
isSend = st.button("Send")

# Check if send button is clicked or not
if isSend:
    st.session_state.past.append(user_input)
    msg = main_(user_input.lower())
    index = msg[1]
    question = msg[2]
    st.session_state.ind = index
    st.session_state.question = question
    st.session_state.generated.append(msg[0])

# Show all messages that user asks and the responses that the user gets
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
