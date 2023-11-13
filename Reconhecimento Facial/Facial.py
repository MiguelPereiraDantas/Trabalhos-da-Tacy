import dlib
import cv2
import numpy as np

# Carregar o modelo pré-treinado para detecção facial
detector = dlib.get_frontal_face_detector()

# Carregar o modelo pré-treinado para pontos de referência faciais (shape predictor)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Carregar uma imagem de exemplo
imagem = cv2.imread("exemplo.jpg")

# Converter a imagem para escala de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar faces na imagem
faces = detector(imagem_gray)

# Iterar sobre as faces detectadas
for face in faces:
    # Obter pontos de referência faciais
    landmarks = predictor(imagem_gray, face)
    
    # Desenhar um retângulo ao redor da face
    cv2.rectangle(imagem, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    
    # Iterar sobre os pontos de referência faciais e desenhá-los na imagem
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(imagem, (x, y), 2, (255, 0, 0), -1)

# Exibir a imagem com as faces e pontos de referência
cv2.imshow("Reconhecimento Facial", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()