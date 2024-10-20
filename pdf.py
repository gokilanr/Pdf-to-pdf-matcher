import streamlit as st
import pdfplumber
from google.cloud import aiplatform
from google.auth import default
import streamlit as st
from PyPDF2 import PdfReader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from google.cloud import aiplatform
from io import BytesIO

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Google Cloud Vertex AI Setup for Gemini
PROJECT_ID = "adept-lodge-427916-r3"
LOCATION = "asia-south1"  # Adjust based on your region
MODEL_ID = "text-bison"  # Gemini model's identifier in Vertex AI

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Function to call Generative AI API for text extraction or manipulation
def extract_fees_with_generative_ai(text):
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    
    # Directly pass the text to the model's predict method
    response = model.predict(text)  # 'text' should be passed directly here

    # Assuming the response contains the extracted/generative text
    return response


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract specific fees from the extracted text based on fee categories
def extract_fees_by_category(text):
    fees = {
        'A. Origination Charges': '',
        'B. Services Borrower Did Not Shop For': '',
        'C. Services Borrower Did Shop For': '',
        'E. Taxes and Other Government Fees': ''
    }

    # Extract relevant sections based on simple keyword matching
    for category in fees:
        start_index = text.find(category)
        if start_index != -1:
            # Extract until the next fee category or a new section
            end_index = min([text.find(next_category, start_index) for next_category in fees if next_category != category and text.find(next_category, start_index) != -1] + [len(text)])
            fees[category] = text[start_index:end_index].strip()

    return fees

# Function to match fees between Invoice and ALTA Statement
def match_fees(invoice_fees, alta_fees):
    matching_fees = []
    mismatching_fees = []

    # Compare the fees from both documents
    for category, fee in invoice_fees.items():
        if category in alta_fees and fee == alta_fees[category]:
            matching_fees.append((category, fee))
        else:
            mismatching_fees.append((category, fee, alta_fees.get(category, 'Not in ALTA')))

    return matching_fees, mismatching_fees

# Streamlit interface
st.title("Fee Matcher for Closing Disclosure and ALTA Statement")

st.write("Upload the Closing disclosure and ALTA Statement PDFs to compare fees (A, B, C, E).")

# Upload PDFs
invoice_pdf = st.file_uploader("Upload Closing disclosure PDF", type="pdf")
alta_pdf = st.file_uploader("Upload ALTA Statement PDF", type="pdf")

if invoice_pdf and alta_pdf:
    # Extract text from PDFs
    invoice_text = extract_text_from_pdf(invoice_pdf)
    alta_text = extract_text_from_pdf(alta_pdf)

    # Extract specific fees by categories
    invoice_fees = extract_fees_by_category(invoice_text)
    alta_fees = extract_fees_by_category(alta_text)

    # Match and highlight fees
    matching_fees, mismatching_fees = match_fees(invoice_fees, alta_fees)

    # Prepare the data for display and Excel output
    matching_data = {"Category": [item[0] for item in matching_fees], "Fee (Invoice)": [item[1] for item in matching_fees], "Fee (ALTA)": [item[1] for item in matching_fees]}
    mismatching_data = {"Category": [item[0] for item in mismatching_fees], "Fee (Invoice)": [item[1] for item in mismatching_fees], "Fee (ALTA)": [item[2] for item in mismatching_fees]}

    df_matching = pd.DataFrame(matching_data)
    df_mismatching = pd.DataFrame(mismatching_data)

    # Display matching and mismatching fees
    st.subheader("Matching Fees")
    st.write(df_matching)

    st.subheader("Mismatching Fees")
    st.write(df_mismatching)

    # Allow the user to download the comparison as an Excel file
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Fee Comparison')
        processed_data = output.getvalue()
        return processed_data

    if st.button("Download Comparison as Excel"):
        excel_data_matching = to_excel(pd.concat([df_matching, df_mismatching]))
        st.download_button(
            label="Download Excel",
            data=excel_data_matching,
            file_name="fee_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
