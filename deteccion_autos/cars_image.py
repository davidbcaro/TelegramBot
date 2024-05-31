import cv2  # Importa la biblioteca OpenCV

# Cargar la imagen
image = cv2.imread("cars1.png")  # Carga el archivo de imagen

# Cargar el clasificador en cascada para la detección de autos
carros = cv2.CascadeClassifier("haarcascade_car.xml")

# Convertir la imagen a escala de grises para mejorar la identificación
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar autos en la imagen
cars = carros.detectMultiScale(gray, 1.1, 1)

# Dibujar un rectángulo alrededor de cada auto detectado
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con los autos detectados
cv2.imshow("Autos Detectados", image)
cv2.waitKey(0)  # Esperar hasta que se presione una tecla

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()


