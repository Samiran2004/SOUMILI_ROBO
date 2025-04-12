import random
import time
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 145)

# Set female voice if available
voices = engine.getProperty('voices')
for voice in voices:
    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load multilingual emotion detection model (EmoRoBERTa)
emotion_model_name = "nateraw/bert-base-uncased-emotion"

# Load tokenizer and model (PyTorch version, no TF, no config changes)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model.eval()  # Set model to evaluation mode

# EmoRoBERTa emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Enable translation if needed
USE_TRANSLATION = True

def detect_emotion(text):
    print(f"[DEBUG] Original text: {text}")

    # Optionally translate text to English
    if USE_TRANSLATION:
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            print(f"[DEBUG] Translated text: {translated_text}")
        except Exception as e:
            print(f"[DEBUG] Translation failed: {e}")
            translated_text = text
    else:
        translated_text = text

    inputs = tokenizer(translated_text, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    print(f"[DEBUG] Emotion probabilities: {probabilities}")
    top_emotion = torch.argmax(probabilities).item()
    return emotion_labels[top_emotion]

# Sample responses
responses = {
    'joy': ["Aww, you're so happy! That makes me smile too!", "Yay! I'm happy because you are."],
    'sadness': ["Oh no, I'm here for you. Want to talk about it?", "I'm sending you a virtual hug."],
    'anger': ["Take a deep breath... I'm here to listen.", "Let's calm down together, okay?"],
    'fear': ["You're safe with me. I'm not going anywhere.", "I understand, but don't worry – I'm here."],
    'disgust': ["Yuck! What happened? Tell me more.", "That doesn't sound pleasant. I'm here if you need to vent."],
    'surprise': ["Whoa! That's unexpected. Tell me more!", "Surprises can be fun, or scary! What kind is it?"],
    'neutral': ["I'm all ears. What's on your mind?", "Tell me anything you like – I'm listening."]
}

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio, language="hi-IN")
            print("You said:", query)
            return query
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError:
            print("Request error. Please check your internet.")
            return ""

def chat():
    print("Hey! I'm your AI GF 🤖💖 Talk to me in Hindi or English. Say 'bye' to exit.")
    speak("Hi baby! I'm your AI girlfriend. Talk to me!")
    while True:
        user_input = get_voice_input()
        if not user_input:
            continue
        if 'bye' in user_input.lower() or 'बाय' in user_input.lower():
            goodbye = "Bye love! Take care 💕"
            print(goodbye)
            speak(goodbye)
            break
        emotion = detect_emotion(user_input)
        print(f"Detected emotion: {emotion}")
        reply = random.choice(responses.get(emotion, ["Hmm, I'm listening. Tell me more..."]))
        print("AI GF:", reply)
        speak(reply)

if __name__ == "__main__":
    chat()
