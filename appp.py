import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorization.pkll','rb'))
model = pickle.load(open('model.pkll','rb'))


st.title("email/sms spam classifier")

input_sms = st.text_area("enter the message")
if st.button('predict'):

    # 1.preprocess
    transform_sms = transform_text(input_sms)
    # 2.vector
    # 2.vector
    vector_input = tfidf.transform([transform_sms])
    # 3.predict
    result = model.predict(vector_input)[0]

    # 4.display
    if result == 1:
        st.header("spam")

    else:
        st.header("not spam")