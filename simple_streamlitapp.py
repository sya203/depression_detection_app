#created by hasya farwizah
#import necessary libraries

#from textblob import TextBlob
import numpy as np
import streamlit as st
import pickle
import joblib,os
import pandas as pd
#import cleantext
#NLP Pkgs
import spacy
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
#Wordcloud
from wordcloud import WordCloud, ImageColorGenerator

#vectorizer
text_vectorizer = open("tfidf.pkl","rb")
posts_cv = joblib.load(text_vectorizer)

#Load prediction model
def load_model(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
        
#main function
def main():
    #""Depression detection app with streamlit""
    st.title("Depression Detection through Textual Social Media posts using ML")
    st.subheader("NLP and ML App with Streamlit")

    activities=["Prediction","NLP"]
    choice = st.sidebar.selectbox("Choose Activity",activities)

    if choice == 'Prediction':
        st.info("Prediction with ML")
        post_text = st.text_area("Enter Text", "Type Here")
        prediction_labels = {'normal':0, 'depression':1}
        if st.button("Classifier"):
            #st.text("Original test::{}\n".format(post_text))
            st.text("{}\n".format(post_text))
            vect_text = posts_cv.transform([post_text]).toarray()
            predictor = load_model("/Users/hasyafarwizah/Downloads/rapidfinalmodel.pkl")
            prediction = predictor.predict(vect_text)
            #st.write(prediction)
            final_result = get_keys(prediction,prediction_labels)
            st.success("This sentence contains {} signs".format(final_result))


    if choice == 'NLP':
        st.info("Natural Language Processing of Text")
        raw_text = st.text_area("Enter Text Here","Type Here")
        nlp_task = ["Tokenization","Lemmatization","POS Tags"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button("Analyze"):
            st.info("Original Text::\n{}".format(raw_text))
            docx = nlp(raw_text)
            if task_choice == 'Tokenization':
                result = [token.text for token in docx ]
            elif task_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

            st.json(result)

        if st.button("Tabulize"):
            docx = nlp(raw_text)
            c_tokens = [token.text for token in docx ]
            c_lemma = [token.lemma_ for token in docx ]
            c_pos = [token.pos_ for token in docx ]

            new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
            st.dataframe(new_df)


    if st.checkbox("WordCloud"):
        c_text = raw_text
        wordcloud = WordCloud().generate(c_text)
        plt.imshow(wordcloud,interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__ == '__main__':
    main()
