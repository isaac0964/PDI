# Importar liibrerias requeridas
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Capturar video
cap = cv.VideoCapture(0)
# Ajustar tamano de la img
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

# Crear trackbar tamano filtro box
def n(x):
    pass

def sigma(x):
    pass

cv.namedWindow("box")
cv.namedWindow("gaussiano")
cv.createTrackbar("n", "box", 1, 15, n)
cv.createTrackbar("sigma", "gaussiano", 1, 10, sigma)

print(cv.getGaussianKernel(5, 0))
while True:
    # Obtener img de la webcam
    _, src = cap.read()
    cv.imshow("Imagen BGR", src)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("Imagen escala grises", src)

    # Filtros de suavizado
    n = cv.getTrackbarPos("n", "box")  # Tamano del filtro
    sigma = cv.getTrackbarPos("sigma", "gaussiano")
    box = np.ones((n, n)) / (n * n)
    img_gauss = cv.GaussianBlur(src, (7,7), sigma)
    # filtrar img
    img_box = cv.filter2D(src, -1, box)

    # Filtros de bordes
    # Sobel
    hx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    hy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    sx = cv.filter2D(img_box, -1, hx)
    sy = cv.filter2D(img_box, -1, hy)
    s = cv.addWeighted(sx, 0.5, sy, 0.5, 0)
    s = cv.inRange(s, 50, 255)

    # Laplace
    hl = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])
    
    l = cv.filter2D(img_box, -1, hl)
    l = cv.inRange(l, 60, 255)
    
    cv.imshow("box", img_box)
    cv.imshow("gaussiano", img_gauss)
    cv.imshow("Sobel", s)
    cv.imshow("Laplace", l)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

