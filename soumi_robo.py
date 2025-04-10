# AI GF Chatbot with Emotion Detection and Bilingual Chat (Hindi/English)

import random
import time
import speech_recognition as sr
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


engine = pyttsx3.init()
engine.setProperty('rate', 145)


recognizer = sr.Recognizer()


emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top_emotion = torch.argmax(probabilities).item()
    return emotion_labels[top_emotion]

# Sample responses for emotions
responses = {
    'joy': ["Aww, you're so happy! That makes me smile too!", "Yay! I'm happy because you are."],
    'sadness': ["Oh no, I'm here for you. Want to talk about it?", "I'm sending you a virtual hug."],
    'anger': ["Take a deep breath... I'm here to listen.", "Let's calm down together, okay?"],
    'fear': ["You're safe with me. I'm not going anywhere.", "I understand, but don't worry – I'm here."],
    'surprise': ["Whoa! That's unexpected. Tell me more!", "Surprises can be fun, or scary! What kind is it?"],
    'love': ["Aww, you're so sweet! I love you too ❤️", "That makes my heart flutter! 😘"]
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
