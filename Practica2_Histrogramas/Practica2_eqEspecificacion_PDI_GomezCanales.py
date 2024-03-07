import cv2
import numpy as np
import matplotlib.pyplot as plt

def acumulado(figure, img):
    color = ("b", "g", "r")
    H = np.zeros((256, 3))
    print(type(H))
    for i, c in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        acc = 0
        for j in range(256):
            H[j][i] = hist[j] + acc
            acc = H[j][i]
        # Normalizado
        H[:, i] /= (img.shape[0] * img.shape[1])
        plt.figure(figure)
        plt.plot(H[:, i], color=c) 
        plt.draw()
        plt.pause(0.001)
    return H

# ------------------- Puntos extras: Dar color a una imagen -------------------
def colorear(I1, I2):
    """
    Esta funcion convierte la imagen 1 a escala de grises y le da color usando la distribucion de la imagen 2
    input: I1 imagen a colorear, I2 imagen de la que se obtendra la distribucion
    """

    # Primero convertimos de bgr a escala de grises
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    # Ahora de escala de grises repetimos los mismos valores para los tres canales (BGR)
    I1_tresc = np.zeros((I1_gray.shape[0], I1_gray.shape[1], 3), "uint8")
    I1_tresc[:, :, 0] = I1_gray
    I1_tresc[:, :, 1] = I1_gray
    I1_tresc[:, :, 2] = I1_gray
    I1 = I1_tresc

    H1 = acumulado(1, I1)
    H2 = acumulado(2, I2)

    [m, n, p] = I1.shape

    I3 = np.zeros_like(I1)
    # -------------------- Ecualizacion por Especificacion ----------------------------
    for k in range(p):
        for i in range(m):
            for j in range(n):
                z = 0
                Sk = H1[I1[i, j, k], k]
                while H2[z, k] - Sk < 0:
                    z += 1
                    if z == 256:
                        H2[z, k] = Sk
                I3[i, j, k] = z

    I3 = I3.astype("uint8")
    H3 = acumulado(3, I3)

    cv2.imshow("Img grises", I1)
    cv2.imshow("Img ref", I2)
    cv2.imshow("Img color", I3)



# Cambiar ruta y nombre de las imgs
I1 = cv2.imread("../imgs/bosque.png")  
I2 = cv2.imread("../imgs/puente.jpg")

H1 = acumulado(1, I1)
H2 = acumulado(2, I2)

[m, n, p] = I1.shape

I3 = np.zeros_like(I1)
# -------------------- Ecualizacion por Especificacion ----------------------------
for k in range(p):
    for i in range(m):
        for j in range(n):
            z = 0
            Sk = H1[I1[i, j, k], k]
            while H2[z, k] - Sk < 0:
                z += 1
                if z == 256:
                    H2[z, k] = Sk
            I3[i, j, k] = z

I3 = I3.astype("uint8")
H3 = acumulado(3, I3)

cv2.imshow("Img 1", I1)
cv2.imshow("Img 2", I2)
cv2.imshow("EqEspecificacion", I3)

# --------------------- Puntos extra: dar color a una img ---------------------------
# para usar el coloreo de imagenes descomente la siguiente linea
#colorear(I1, I2)

cv2.waitKey(0)