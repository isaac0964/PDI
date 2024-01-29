import cv2
import numpy as np
import matplotlib.pyplot as plt

# Capturar video
cap = cv2.VideoCapture(0)
# Ajustar tamano de la img
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def plothist(figure, img):
    plt.figure(figure)
    plt.clf()
    # Los parametros de calcHist son:
    # lista con imagenes a las que se le calculara el hist, canales que se usaran para calcular el hist,
    # mascara opcional del mismo tamano del a img, tamano del hist en cada dimension, rango del hist
    gray_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.draw()
    plt.pause(0.01)

def acumulado(figure, img):
    gray_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    acc = 0
    H = np.zeros(256)
    for i in range(256):
        H[i] = gray_hist[i] + acc
        acc = H[i]
    plt.figure(figure)
    plt.clf()
    plt.plot(H)
    plt.draw()
    plt.pause(0.001)
    return H

# Alpha blend
def alpha(x):
    pass

cv2.namedWindow('alphablend')
cv2.createTrackbar('alpha', 'alphablend', 0, 100, alpha)

while True:
    _, src = cap.read()
    cv2.imshow("Imagen BGR", src)

    # -------- Desplegar Histograma -------------
    """
    plt.clf()
    color = ("b", "g", "r")
    for i, c in enumerate(color):
        histr = cv2.calcHist([src], [i], None, [256], [0, 256])
        plt.plot(histr, color=c)
        plt.xlim([0, 256])
    plt.draw()
    plt.pause(0.01)
    """

    # -------- Contraste brillo -------------
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    graymod = gray * 0.3 + 80
    graymod = graymod.astype("uint8")

    plothist(0, gray)
    plothist(1, graymod)

    cv2.imshow("Gray", gray)
    cv2.imshow("Gray_mod", graymod)
    """
    # -------- Adaptacion Automatica -------------
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    graymod = gray * 0.3 + 80
    graymod = graymod.astype("uint8")

    bajo = graymod.min()
    alto = graymod.max()
    grayad = np.zeros_like(gray, np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            grayad[i][j] = ((graymod[i][j] - bajo) / (alto - bajo)) * 255
    grayad = grayad.astype("uint8")

    plothist(0, graymod)
    plothist(1, grayad)
    cv2.imshow("Gray mod", graymod)
    cv2.imshow("Adapatacion automatica", grayad)
    """

    # -------- Ecualizacion Lineal -------------
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    H = acumulado(1, gray)

    # Ecualizacion
    Ieq = np.zeros_like(gray, np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            v = gray[i][j]
            Ieq[i][j] = H[v] * (255 / (gray.shape[0] * gray.shape[1]))

    # Acumulado ecualizado
    acumulado(2, Ieq)
    cv2.imshow("Ecualizacion Lineal", Ieq)
    cv2.imshow("Imagen Original", gray)
    """

    # -------- Alpha Blending -------------
    img = cv2.imread("../imgs/lena.png")  # Cambiar la ruta a donde se tenga guardada la img
    img = cv2.resize(img, (src.shape[1], src.shape[0]))  # Primero ancho y despues alto, ancho es la dim 1 y alto la dim 0ape, src)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    alpha_value = (cv2.getTrackbarPos("alpha", "alphablend")) / 100

    # Los parametros de addWeighted son: 
    # src1, alpha, src2, beta (peso de src2), gama (escalar que se añade a la suma)
    outImg = cv2.addWeighted(img, alpha_value, src_gray, 1-alpha_value, 0)  # El error es que se estaba haciendo con la imagen a color en vez de la src_gray
    cv2.imshow("alphablend", outImg)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break