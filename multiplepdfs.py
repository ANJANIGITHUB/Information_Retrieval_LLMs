#Chatbot for Information Retrival

# Name                 : Anjani Kumar
# BITS ID              : 2021SC04645 
# Date                 : 09/12/2023  
# Dissertation Title   : Information Retrieval Chat Bot Using Generative AI & LLMs

#Import all the required libraries

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import nltk
from nltk.translate.bleu_score import sentence_bleu
#from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import spacy
from spacy import displacy
import os
from wordcloud import WordCloud

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from time import perf_counter
import numpy as np

#import comparepdfs

#Function to read the uploaded PDF Documents
def get_text_from_pdfs(pdf_documents):
    text=""
    for pdf in pdf_documents:
        pdf_reader= PdfReader(pdf)

        num_pages = len(pdf_reader.pages)
        st.write(f"1. Number of pages in the PDF are {num_pages}")

        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#Function to read the text chunks recursively
def get_text_chunks_from_pdfs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#Function to store the text chunks as vector embeddings using FAISS(Facebook AI Similarity Search) vector db 
# and save it locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#Function to start the conversation using Prompt
def start_conversation():

    prompt_template = """You are expert in reading and understanding PDF documents.Read the uploaded documents
    carefully and answer the questions asked by users.Please do not give wrong answers\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


#Function to handle user questions and return response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = start_conversation()

    response = chain({"input_documents":docs, "question": user_question}, 
                     return_only_outputs=True
                    )
    
    # Get the Named - Entities
    #st.write("4. Named and Associated Entities are :")
    get_named_entity(response["output_text"])

    # Write Response on Page
    # st.subheader("Response :")
    # st.write(response["output_text"])
    

    return response["output_text"]


#Functions for Performing Preprocessing with the data

def preprocess_pdf_text(text):
    # Remove unwanted characters (non-alphabetic and non-space)
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Split into words
    words = cleaned_text.split()
    #Stop Word Removal
    stop_words = set(stopwords.words('english'))
    words = [lemma.lemmatize(word) for word in words if word.lower() not in stop_words]
            
    return words

# Function to count words
def count_words(words):
    return len(words)

NER = spacy.load("en_core_web_sm")

#Get Name and associated Entities
def get_named_entity(raw_text):
  text2 = NER(raw_text)

  # Extract named entities
  entities = [(ent.text, ent.label_) for ent in text2.ents]

  # Create DataFrame for named entities
  entities_df = pd.DataFrame(entities, columns=['Entity', 'Label'])

  st.write("4.Named Entities:")
  st.write(entities_df)
    
  # for word in text2.ents:
  #   st.write(word.text,'-',word.label_)

#initialize WordNetLemmatizer
lemma=WordNetLemmatizer()

#Performance Evaluation using BLEU Score
def calculate_bleu_score(reference, candidate):
    # Tokenize the reference and candidate sentences
    
    reference = re.sub(r'[^a-zA-Z0-9\s]', '', reference)
    candidate = re.sub(r'[^a-zA-Z0-9\s]', '', candidate)
    
    reference_tokens = [word.lower() for word in nltk.word_tokenize(reference)]
    candidate_tokens = [word.lower() for word in nltk.word_tokenize(candidate)]

    stop_words = set(stopwords.words('english'))
    reference_tokens = [word for word in reference_tokens if word.lower() not in stop_words]
    candidate_tokens = [word for word in candidate_tokens if word.lower() not in stop_words]
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)

    return round(bleu_score,2)

#Questions for Performance Calculations
bleuscore=0
question1="Who are the founders of Maersk?"
question2="Which years did Regulation violations and contract fraud happened?"
question3="What all are the Maersk products?"

#Define main function
def main():

    #Load Environments and API Key
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
   
    st.header("Chat with multiple PDFs :books:")
    
    #Application Information on the side bar of the streamlit page
    with st.sidebar:
      #st.title('QnA LLM Chat App 💬')
      st.markdown('''
      ## About
     This LLM QnA ChatBot app is built using:
    - [LangChain Framework](https://python.langchain.com/)
    - [Google Generative AI](https://ai.google/discover/generativeai/)
    - [FAISS Vector DB](https://ai.meta.com/tools/faiss/)
    - [Streamlit App](https://streamlit.io/)
    - Ask Any Questions related to your uploaded PDFs and you will get your answer.
    ''')
      st.write('Author: Anjani Kumar')

    pdf_docs = st.sidebar.file_uploader("Upload your PDFs here and click on 'Submit'", accept_multiple_files=True)
    user_question = st.text_input("Ask any question about your PDFs:")

    if st.button("Submit"):
        with st.spinner("Processing"):
        
          st.subheader("Uploaded PDF General Information:")
          # get the text chunks

          if pdf_docs:
            raw_text = get_text_from_pdfs(pdf_docs)
          else:
            st.experimental_rerun()

        #   raw_text = get_pdf_text(pdf_docs)

        #Preprocessing Start

        # Data Exploration section

          preprocessed_words = preprocess_pdf_text(raw_text)
          word_count         = count_words(preprocessed_words)
          st.write(f"2. Total Word Count from PDF is {word_count}")

         # Calculate word frequency
          freq_dist = FreqDist(preprocessed_words)
    
         # Get the most common words
          frequent_words = freq_dist.most_common(10)
          st.write(f"3. Frequent words from pdf are \n\n{frequent_words}")
                   
          text_chunks = get_text_chunks_from_pdfs(raw_text)
          get_vector_store(text_chunks)

          if user_question:
            try:
                # Performance Counter
                tick = perf_counter()

                response=user_input(user_question)
                st.subheader("Response :")
                st.write(response)

                total_time=perf_counter() - tick
                st.subheader(f"Response Time is {np.round(total_time,3)} Seconds")
            except Exception:
                import traceback
                st.write(":red[Error in processing user_input...]")
                st.write(traceback.format_exc())
          st.success("Done")

          #Preprocessing End
         
        #Only for Performance Evaluation.Should be commented later:Start
          
          if bleuscore==1 and user_question==question1:
            actual_answer = "The Founders of Maersk are Arnold Peter Møller and Peter Mærsk Møller"
            bleu_score = calculate_bleu_score(actual_answer, response)
            st.subheader("Bleu Score")
            st.write(f"Bleu Score for this question is {bleu_score}")

          elif bleuscore==1 and user_question==question2:
            actual_answer = "2010 and 2014"
            bleu_score = calculate_bleu_score(actual_answer, response)
            st.subheader("Bleu Score")
            st.write(f"Bleu Score for this question is {bleu_score}")

          elif bleuscore==1 and user_question==question3:
            actual_answer = """Maerk Products are Container shipping and terminals, logistics and
                                    freight forwarding, ferry and tanker transport,
                                    semi-submersible drilling rigs and FPSOs"""
            bleu_score = calculate_bleu_score(actual_answer, response)
            st.subheader("Bleu Score")
            st.write(f"Bleu Score for this question is {bleu_score}")

          #Only for Performance Evaluation.Should be commented later:End


if __name__ == '__main__':
    main()
