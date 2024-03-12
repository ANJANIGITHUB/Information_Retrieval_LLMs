#Chatbot for Information Retrival

# Name                 : Anjani Kumar
# BITS ID              : 2021SC04645 
# Date                 : 09/12/2023  
# Dissertation Title   : Information Retrieval Chat Bot Using Generative AI & LLMs

#Import all the required libraries

import streamlit as st
import PyPDF2
import openai
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from time import perf_counter
import numpy as np


#Define all the required function for reading PDFs and comparing
def process_pdf(uploaded_file):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        #st.subheader(f"pdf {uploaded_file.name} Basic Information")
        #st.write("*************************************")
        st.write(f"Number of pages in the PDF {uploaded_file.name} is {num_pages}")

        #st.subheader("Text extracted from the PDF:")
        text = ''
        for page_num in range(num_pages):
            text += pdf_reader.pages[page_num].extract_text()
        #return(text.extract_text())
        return text


def compare_pdf_content(pdf1_text, pdf2_text):
    #prompt = f"""Compare the content of two PDFs and share the few Similarities and Differences in the documents:\n\nPDF 1:\n{pdf1_text}\n\nPDF 2:\n{pdf2_text}"""
    prompt = f"""Compare the content of two PDFs and share the Differences from both the documents:\n\nPDF 1:\n{pdf1_text}\n\nPDF 2:\n{pdf2_text}"""
    
    #load_dotenv()
    api_key=st.secrets["OPENAI_API_KEY"]
    # Set up your OpenAI API key
    openai.api_key = api_key
    
    # Make a request to GPT-3
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    comparison_result = response.choices[0].text.strip()
    return comparison_result

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

#initialize WordNetLemmatizer
lemma=WordNetLemmatizer()

def main():

    #load_dotenv()
    # st.set_page_config(page_title="Compare two PDFs",
    #                    page_icon=":books:")
    #st.write(css, unsafe_allow_html=True)
    st.header("   Compare two PDF files:books:")

    with st.sidebar:
      #st.title('QnA LLM Chat App 💬')
      st.markdown('''
      ## About
     This PDFs Comparison app is built using:
    - [OpenAI](https://platform.openai.com/docs/models)
    - [Streamlit](https://streamlit.io/)
    - This compares two uploaded PDFs.
    ''')
      add_vertical_space(2)
      st.write('Author: Anjani Kumar')

    uploaded_file1 = st.file_uploader("Upload a PDF1 file", type=["pdf"])
    uploaded_file2 = st.file_uploader("Upload a PDF2 file", type=["pdf"])

    if st.button("Compare"):
        with st.spinner("Processing"):
         
         st.subheader(f"Uploaded PDFs Summary")

         if uploaded_file1 is not None:
            st.write(f"PDF 1, name {uploaded_file1.name} uploaded Sucessfully")

         if uploaded_file2 is not None:
             st.write(f"PDF 2, name {uploaded_file2.name} uploaded Sucessfully")
             
         st.write("*************************************")
         pdf1_text=process_pdf(uploaded_file1)
         pdf2_text=process_pdf(uploaded_file2)

         # Data Exploration section

         #pdf1_text           = process_pdf(uploaded_file1)
         preprocessed_words1 = preprocess_pdf_text(pdf1_text)
         word_count1         = count_words(preprocessed_words1)

         #pdf2_text           = process_pdf(uploaded_file2)
         preprocessed_words2 = preprocess_pdf_text(pdf2_text)
         word_count2         = count_words(preprocessed_words2)


         st.write("*************************************")
         st.write(f"Total Word Count for {uploaded_file1.name} is {word_count1}")
         st.write(f"Total Word Count for {uploaded_file2.name} is {word_count2}")

         # Calculate word frequency
         freq_dist1 = FreqDist(preprocessed_words1)
         freq_dist2 = FreqDist(preprocessed_words2)
        #  st.write(freq_dist)
    
        # Get the most common words
         frequent_words1 = freq_dist1.most_common(5)
         frequent_words2 = freq_dist2.most_common(5)
         
         st.write("*************************************")
         st.write(f"Frequent words from pdf {uploaded_file1.name} is \n\n{frequent_words1}")
         st.write(f"Frequent words from pdf {uploaded_file2.name} is \n\n{frequent_words2}")


         # Compare the content using GPT-3
         # Performance Counter
         tick = perf_counter()
         try:
             result = compare_pdf_content(pdf1_text, pdf2_text)

            # Showthe comparison result
             st.subheader("**Comparison Result:**")
             st.write(result)
             total_time=perf_counter() - tick
             st.subheader(f"Time span for Comparing PDFs is {np.round(total_time,3)} Seconds")
         except Exception:
                import traceback
                st.write(":red[Error in PDFs Comparison] ")
                st.write(traceback.format_exc())
                
        
         
    


if __name__ == "__main__":
    main()
