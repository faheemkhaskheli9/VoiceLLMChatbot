import pyttsx3
import speech_recognition as sr

from chatbot import ChatBot


class VoiceChatbot:
    def __init__(self, name):
        self.name = name
        self.chatbot = ChatBot()

        # Set up the text-to-speech engine
        self.engine = pyttsx3.init()

        self.voice_recognizer = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            audio = self.voice_recognizer.listen(source, phrase_time_limit=5)
            print("Processing...")
        try:
            text = self.voice_recognizer.recognize_google(audio)
            return text
        except Exception as e:
            print("Error: " + str(e))
        return None

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def run(self):
        self.chatbot.ask(
            f"You are an assistant for question-answering tasks. Use three sentences maximum and keep the answer concise. Your name is {self.name}."
        )
        self.speak(f"Hello, I am {self.name}. How can I help you today?")

        while True:
            prompt = self.listen()
            if prompt is not None:
                print("You: " + prompt)

                response = self.chatbot.ask(prompt)
                print(self.name, ":", response)

                if "Have a great day!" in response:
                    exit()

                self.speak(response)
            else:
                self.speak("I'm sorry, I didn't understand that.")


if __name__ == "__main__":
    bot = VoiceChatbot("Faheem Bot")
    bot.run()
