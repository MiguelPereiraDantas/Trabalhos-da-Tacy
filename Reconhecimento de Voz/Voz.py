import speech_recognition as sr

# Inicializar o reconhecedor
recognizer = sr.Recognizer()

# Capturar áudio do microfone
with sr.Microphone() as source:
    print("Fale algo...")
    audio = recognizer.listen(source)

# Tentar reconhecer o texto do áudio
try:
    print("Texto reconhecido: " + recognizer.recognize_google(audio))
except sr.UnknownValueError:
    print("Não foi possível reconhecer o áudio.")
except sr.RequestError as e:
    print(f"Erro na solicitação ao Google API; {e}")

