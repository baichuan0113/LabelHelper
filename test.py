# import speech_recognition
# import pyttsx3
# import pyaudio

# recognizer = speech_recognition.Recognizer()

# while True:
#     try: 
#         with speech_recognition.Microphone() as mic:
#             recognizer.adjust_for_ambient_noise(mic, duration=0.2)
#             audio = recognizer.listen(mic)

#             text = recognizer.recognizer_google(audio)
#             text = text.lower()

#             print(f"Recognized {text}")

#     except speech_recognition.UnknownValueError():

#         recognizer = speech_recognition.Recognizer()
#         continue
from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))