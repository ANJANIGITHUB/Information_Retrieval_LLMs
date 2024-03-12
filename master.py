#Master python file for Information Retrival Application

# Name                 : Anjani Kumar
# BITS ID              : 2021SC04645 
# Date                 : 04/01/2024 
# Dissertation Title   : Information Retrieval Chat Bot Using Generative AI & LLMs

#Import all the required libraries

# master.py
import streamlit as st
from multiplepdfs import main as multipdfs
from comparepdfs import main as comparepdfs
from PIL import Image
import openai

def set_background(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    # Set background color to transparent
    st.markdown(
        """
        <style>
            .stApp {
                background-color: transparent;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():

    openai.api_key = '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'
    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")\

    # Upload an image from your local folder
    set_background("llm.jpg")

    st.title("Gen AI & LLMs Working Demo")

    app_selection = st.sidebar.selectbox(
        "Choose Application",
        ["Mutiple PDFs QnA", "Compare two PDFs"]
    )

    if app_selection == "Mutiple PDFs QnA":
        multipdfs()
    elif app_selection == "Compare two PDFs":
        comparepdfs()

if __name__ == "__main__":
    main()
