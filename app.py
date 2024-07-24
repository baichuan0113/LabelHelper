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
import av
import time
import pyttsx3
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import asyncio
import threading
import multiprocessing
import sqlite3
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import numpy as np

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

# class AudioRecorder(AudioProcessorBase):
#     def __init__(self):
#         self.frames = []
#         self.recording = False
#         self.q = queue.Queue()
#         self.sr = sr.Recognizer()

#     def recv(self, frame):
#         if self.recording:
#             audio_frame = frame.to_ndarray()
#             self.frames.append(audio_frame)
#             if len(self.frames) * 10 > 48000:  # Roughly 10 seconds of audio at 48kHz
#                 self.recording = False
#                 self.q.put(self.frames.copy())
#                 self.frames = []
#         return frame

#     def start_recording(self):
#         self.frames = []
#         self.recording = True

#     def stop_recording(self):
#         self.recording = False

#     def process_audio(self):
#         while True:
#             frames = self.q.get()
#             audio_data = np.concatenate(frames, axis=0).astype(np.int16)
#             audio_data = audio_data.tobytes()
#             audio_source = sr.AudioData(audio_data, 48000, 2)
#             try:
#                 message = self.sr.recognize_google(audio_source)
#                 st.session_state.recognized_message = message
#                 st.write(f"Recognized Message: {message}")
#                 st.write(f"You spoke {len(message.split())} words.")
#                 result = generate_response(message)
#                 st.session_state['result'] = result
#                 store_message(st.session_state.user_email, message)
#             except sr.UnknownValueError:
#                 st.error("Google Speech Recognition could not understand audio")
#             except sr.RequestError as e:
#                 st.error(f"Could not request results from Google Speech Recognition service; {e}")


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
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

def authenticate_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

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
        else:
            st.error("Invalid email or password")

    st.subheader("Or Sign Up")
    new_email = st.text_input("New Email")
    new_password = st.text_input("New Password", type="password")
    
    if st.button("Sign Up"):
        if register_user(new_email, new_password):
            st.success("Signup successful, you can now login")
        else:
            st.error("Email already registered")

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="response.csv", encoding='iso-8859-1')
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices, 
in terms of length, tone of voice, logical arguments and other details

2/ If the best practices are irrelevant, then try to mimic the style of the best practice to the prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practices of how we normally respond to prospects in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""
prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', 'en-us')
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()


def main():
    st.set_page_config(page_title="Roboflow Labelling Helper", page_icon=":bird:")

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()



def show_main_app():
    st.header(f"Welcome to Roboflow Labelling Helper :bird: ({st.session_state.user_email})")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()

    if 'engine' not in st.session_state:
        st.session_state.engine = pyttsx3.init()
        voices = st.session_state.engine.getProperty('voices')
        st.session_state.engine.setProperty('voice', 'en-us')
        st.session_state.engine.setProperty('rate', 100)

    if 'recognized_message' not in st.session_state:
        st.session_state.recognized_message = ""

    logtxtbox = st.empty()
    logtxtbox.text_area("Recognized Message", value=st.session_state.recognized_message, height=200)



    stt_button = Button(label="Speak", width=100)

    stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
    
        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
            }
        }
        recognition.start();
        """))
    

    temp_msg  = streamlit_bokeh_events(
        stt_button,
        events="GET_TEXT",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)
    if temp_msg:
        if "GET_TEXT" in temp_msg:
            recognized_message = result.get("GET_TEXT")
            st.write(recognized_message)
            st.session_state.recognized_message = recognized_message
            st.write(f"Recognized Message: {recognized_message}")
            st.write(f"You spoke {len(recognized_message.split())} words.")
    # result = generate_response(recognized_message)
    # st.session_state['result'] = result
    # store_message(st.session_state.user_email, recognized_message)

    # recorder = AudioRecorder()
    # webrtc_ctx = webrtc_streamer(
    #     key="example",
    #     mode=WebRtcMode.SENDONLY,
    #     audio_processor_factory=lambda: recorder,
    #     media_stream_constraints={"audio": True},
    # )

    # if st.button('Start Speaking!'):
    #     recorder.start_recording()
    #     st.write("Recording started...")

    #     def stop_recording_after_delay():
    #         threading.Timer(10, stop_recording).start()

    #     def stop_recording():
    #         recorder.stop_recording()
    #         st.write("Recording stopped.")

    #     stop_recording_after_delay()
    #     if not hasattr(st.session_state, 'audio_thread'):
    #         st.session_state.audio_thread = threading.Thread(target=recorder.process_audio, daemon=True)
    #         st.session_state.audio_thread.start()
    
    # Start audio processing thread
    if 'recognized_message' in st.session_state:
        #logtxtbox.text_area("Recognized Message", value=st.session_state.recognized_message, height=200)
        result = generate_response(st.session_state.recognized_message)
        st.session_state['result'] = result
        store_message(st.session_state.user_email, st.session_state.recognized_message)

    if 'result' in st.session_state and st.session_state['result']:
        st.info(st.session_state['result'])
        if st.button('Speak Result'):
            process = multiprocessing.Process(target=speak_text, args=(st.session_state['result'],))
            process.start()
            st.session_state['process'] = process
        if st.button('Stop Speaking'):
            if 'process' in st.session_state:
                st.session_state['process'].terminate()
                st.session_state['process'] = None

    st.subheader("Your Recognized Messages")
    messages = get_messages(st.session_state.user_email)
    for msg, timestamp in messages:
        st.write(f"{timestamp}: {msg}")



    # with sr.Microphone() as source:
        #     calibration_message = st.empty()
        #     calibration_message.write("Please wait. Calibrating microphone...")
        #     r.adjust_for_ambient_noise(source, duration=1)
        #     calibration_message.empty()
        #     speak_message = st.empty()
        #     speak_message.write("Microphone calibrated. Start speaking.")

        #     try:
        #         audio_data = r.listen(source, timeout=10)
        #         speak_message.write("Voice recording completed!")
        #         message = r.recognize_google(audio_data)
        #         st.session_state.recognized_message = message
        #         logtxtbox.text_area("Recognized Message", value=st.session_state.recognized_message, height=200)
        #         speak_message.empty()
        #         st.write(f"You spoke {len(message.split())} words.")
        #         result = generate_response(message)
        #         st.session_state['result'] = result
        #         store_message(st.session_state.user_email, message)
        #     except sr.UnknownValueError:
        #         st.error("Google Speech Recognition could not understand audio")
        #     except sr.RequestError as e:
        #         st.error(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == '__main__':
    main()


# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
# import av
# from langchain_openai import ChatOpenAI
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# import os
# import pyttsx3
# import sqlite3
# import speech_recognition as sr

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Load environment variables
# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY is not set. Please set it in the .env file or in the Streamlit secrets.")

# # Initialize the database
# def init_db():
#     conn = sqlite3.connect('users.db')
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             email TEXT UNIQUE NOT NULL,
#             password TEXT NOT NULL
#         )
#     ''')
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS messages (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id INTEGER NOT NULL,
#             message TEXT NOT NULL,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
#     conn.commit()
#     conn.close()

# init_db()

# def store_message(user_email, message):
#     conn = sqlite3.connect('users.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
#     user_id = cursor.fetchone()[0]
#     cursor.execute('INSERT INTO messages (user_id, message) VALUES (?, ?)', (user_id, message))
#     conn.commit()
#     conn.close()

# def get_messages(user_email):
#     conn = sqlite3.connect('users.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
#     user_id = cursor.fetchone()[0]
#     cursor.execute('SELECT message, timestamp FROM messages WHERE user_id = ?', (user_id,))
#     messages = cursor.fetchall()
#     conn.close()
#     return messages

# def speak_text(text):
#     engine = pyttsx3.init()
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[3].id)
#     engine.setProperty('rate', 100)
#     engine.say(text)
#     engine.runAndWait()

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         with self.microphone as source:
#             audio_data = self.recognizer.listen(source)
#             try:
#                 text = self.recognizer.recognize_google(audio_data)
#                 st.write(f"Recognized: {text}")
#                 store_message(st.session_state.user_email, text)
#                 st.session_state.recognized_message = text
#             except sr.UnknownValueError:
#                 st.write("Google Speech Recognition could not understand audio")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results from Google Speech Recognition service; {e}")
#         return frame

# def main():
#     st.set_page_config(page_title="Roboflow Labelling Helper", page_icon=":bird:")

#     if 'authenticated' not in st.session_state:
#         st.session_state.authenticated = False

#     if not st.session_state.authenticated:
#         show_login_page()
#     else:
#         show_main_app()

# def show_login_page():
#     st.title("Login")
#     email = st.text_input("Email")
#     password = st.text_input("Password", type="password")
    
#     if st.button("Login"):
#         if authenticate_user(email, password):
#             st.session_state.authenticated = True
#             st.session_state.user_email = email
#             st.success("Login successful")
#         else:
#             st.error("Invalid email or password")

#     st.subheader("Or Sign Up")
#     new_email = st.text_input("New Email")
#     new_password = st.text_input("New Password", type="password")
    
#     if st.button("Sign Up"):
#         if register_user(new_email, new_password):
#             st.success("Signup successful, you can now login")
#         else:
#             st.error("Email already registered")

# def show_main_app():
#     st.header(f"Welcome to Roboflow Labelling Helper :bird: ({st.session_state.user_email})")

#     if st.button("Logout"):
#         st.session_state.authenticated = False
#         st.experimental_rerun()

#     if 'engine' not in st.session_state:
#         st.session_state.engine = pyttsx3.init()
#         voices = st.session_state.engine.getProperty('voices')
#         st.session_state.engine.setProperty('voice', voices[3].id)
#         st.session_state.engine.setProperty('rate', 100)

#     if 'recognized_message' not in st.session_state:
#         st.session_state.recognized_message = ""

#     logtxtbox = st.empty()
#     logtxtbox.text_area("Recognized Message", value=st.session_state.recognized_message, height=200)

#     webrtc_ctx = webrtc_streamer(
#         key="audio",
#         mode=WebRtcMode.SENDRECV,
#         audio_processor_factory=AudioProcessor,
#         media_stream_constraints={
#             "audio": True,
#             "video": False,
#         },
#         async_processing=True,
#     )

#     st.subheader("Your Recognized Messages")
#     messages = get_messages(st.session_state.user_email)
#     for msg, timestamp in messages:
#         st.write(f"{timestamp}: {msg}")

# def register_user(email, password):
#     conn = sqlite3.connect('users.db')
#     cursor = conn.cursor()
#     try:
#         cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         conn.close()
#         return False
#     conn.close()
#     return True

# def authenticate_user(email, password):
#     conn = sqlite3.connect('users.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
#     user = cursor.fetchone()
#     conn.close()
#     return user is not None

# def generate_response(message):
#     best_practice = retrieve_info(message)
#     response = chain.run(message=message, best_practice=best_practice)
#     return response

# if __name__ == "__main__":
#     main()
