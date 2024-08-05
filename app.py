import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import speech_recognition as sr
import os
import pyttsx3
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import threading
import sqlite3
from audiorecorder import audiorecorder 
import numpy as np
import io
from pydub import AudioSegment
from langchain.chains import RetrievalQA
import firebase_admin
from firebase_admin import auth, credentials
from firebase_admin.exceptions import FirebaseError
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pyrebase

firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}

if not firebase_admin._apps:
    cred = credentials.Certificate("labelhelper-273ed-04873d334799.json")
    firebase_admin.initialize_app(cred)

firebase = pyrebase.initialize_app(firebase_config)
auth_pyrebase = firebase.auth()

def recognize_audio(audio_segment):
    recognizer = sr.Recognizer()
    # Convert AudioSegment to a bytes-like object
    with io.BytesIO() as audio_buffer:
        audio_segment.export(audio_buffer, format="wav")
        audio_buffer.seek(0)
        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError as e:
        text = ""
        #text = f"Could not request results from Google Speech Recognition service; {e}"
    return text



def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


def store_message(user_email, message):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
    user_id = cursor.fetchone()[0]
    cursor.execute('INSERT INTO messages (user_id, message) VALUES (?, ?)', (user_id, message))
    conn.commit()
    conn.close()

def get_messages(user_email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
    user_id = cursor.fetchone()[0]
    cursor.execute('SELECT message, timestamp FROM messages WHERE user_id = ?', (user_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

def register_user(email, password):
    try:
        auth.sign_in_with_email_and_password(email, password)
        return True
    except:
        return False

def authenticate_user(email, password):
    try:
        user = auth_pyrebase.sign_in_with_email_and_password(email, password)
        return True 
    except Exception as e:
        st.error(f"Failed: {e}")
        return False

def send_custom_email(recipient_email, reset_link):
    sender_email = os.getenv("EMAIL")
    sender_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 465

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Password Reset Request"

    body = f"""
    <p>Click the link to reset your password: 
    <a href="{reset_link}">Reset Password</a></p>
    """
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.set_debuglevel(1)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        st.success(f"A password reset link has been sent to {recipient_email}")
    except Exception as e:
        st.error(f"Failed to send email: {e}")
def send_reset_password_email(email):
    try:
        link = auth.generate_password_reset_link(email)
        send_custom_email(email, link)
    except FirebaseError as e:
        st.error(f"An error occurred: {e}")
    

init_db()
load_dotenv()


def show_login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(email, password):
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.subheader("Or Sign Up")
    new_email = st.text_input("New Email")
    new_password = st.text_input("New Password", type="password")
    
    if st.button("Sign Up"):
        if register_user(new_email, new_password):
            st.success("SignUp successful, you can now login")
            st.balloons()
        else:
            st.error("Email already registered")
    
    st.subheader("Forgot Password?")
    reset_email = st.text_input("Enter your email to reset password")
    
    if st.button("Reset Password"):
        if reset_email:
            send_reset_password_email(reset_email)
        else:
            st.error("Please enter your email")

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="response.csv", encoding='iso-8859-1')
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Options include "stuff", "refine", "map_reduce", "map_rerank"
    retriever=db.as_retriever(),
    metadata={"application_type": "question_answering"}
)
# 4. Retrieval augmented generation
def generate_response(query):
    response = chain.invoke({"query": query})
    return response['result']
    

def main():
    st.set_page_config(page_title="Roboflow Labelling Helper", page_icon=":bird: ({st.session_state.user_email})")

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

def speak_text(engine, text):
    def speak():
        print("speaking!!")
        engine.say(text)
        engine.runAndWait()

    if 'speaker_thread' in st.session_state and 'engine' in st.session_state:
        st.session_state.engine.stop()
        st.session_state.speaker_thread.join()

    st.session_state.speaker_thread = threading.Thread(target=speak)
    st.session_state.speaker_thread.start()


def show_main_app():
    st.header("Welcome to Roboflow Labelling Helper :bird:")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    if 'engine' not in st.session_state:
        st.session_state.engine = pyttsx3.init()
        st.session_state.engine.setProperty('voice', 'en-us')
        st.session_state.engine.setProperty('rate', 150)

    if 'recognized_message' not in st.session_state:
        st.session_state.recognized_message = ""

    if 'result' not in st.session_state:
        st.session_state.result = ""


    # Display the recorder and wait for valid audio data
    img = st.image("icon.png", width=50)
    audio_data = audiorecorder("Record your message", key="audiorecorder")
    if audio_data is not None and audio_data.duration_seconds > 0:
        st.session_state.recognized_message = recognize_audio(audio_data)

    # Text area to display recognized message
    textArea = st.text_area("Recognized Message", value=st.session_state.recognized_message, height=200)

    # Button to clear the input text area
    if st.button("Clear Message"):
        st.session_state.recognized_message = ""
        st.session_state.result = ""
        st.rerun()

    # Only proceed if there is a recognized message that is not empty
    if st.session_state.recognized_message.strip():
        if st.button("Generate Response"):
            st.session_state.result = generate_response(st.session_state.recognized_message)
            speak_text(st.session_state.engine, st.session_state.result)
            st.rerun()

    # Display the result and speak it out
    if st.session_state.result:
        st.info(st.session_state.result)
        store_message(st.session_state.user_email, st.session_state.recognized_message)
        # Button to stop speaking
        #and st.session_state.speaker_thread.is_alive()
        if st.button("Stop Speaking"):
            if 'speaker_thread' in st.session_state:
                print("ending2!!!")
                st.session_state.engine.endLoop()
                st.session_state.engine.stop()
                st.session_state.speaker_thread.join()
                st.rerun()

    st.subheader("Your Recognized Messages")
    messages = get_messages(st.session_state.user_email)
    for msg, timestamp in messages:
        st.write(f"{timestamp}: {msg}")

if __name__ == '__main__':
    main()
