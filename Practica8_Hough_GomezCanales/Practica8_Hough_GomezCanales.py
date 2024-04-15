# Angel Isaac Gomez Canales
# Codigo basado en: https://github.com/Jegovila/cursoVR/blob/main/Practica8%3A%20Transformada%20de%20Hough/python/Práctica8.ipynb
# Transformada de Hough 15/04/2024

from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def transformadaHough(img):
    """
    Esta funcion calcula la transormada de Hough de la imagen dada,
    muestra la imagen, la transformada y resalta los picos de la transformada
    input: imagen a la que se le desea calcular la transformada de Hough (binaria)
    """

    # Calcular Transformada de Hough
    h, t, d = hough_line(img)

    # Obtener picos de la transformada
    _, t_peak, d_peak = hough_line_peaks(h, t, d)

    # Mostrar imagen
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
    ax[0].set_title("Imagen Original (Binaria)")
    ax[0].imshow(img, cmap="gray")
    
    # Mostrar la transformadad de Hough de la imagen con los picos resaltados
    ax[1].set_title("Transformada de Hough")
    ax[1].imshow(np.log1p(h), extent=[np.rad2deg(t[0]), np.rad2deg(t[-1]), d[-1], d[0]], cmap="gray", aspect="auto")
    ax[1].set_xlabel("Angulo (grados)")
    ax[1].set_ylabel("Distancia (pixeles)")

    # Mostrar las lineas detectadas
    ax[2].imshow(img, cmap="gray")
    row1, col1 = img.shape

    # Resaltar cada pico
    for theta, dist in zip(t_peak, d_peak):
        # Crear rectangulo
        ancho = 5
        alto = 5 
        ax[1].scatter(np.rad2deg(theta), dist, s=5, c="red")
        #rect = patches.Rectangle((np.rad2deg(theta) - ancho // 2, dist - alto // 2), ancho, alto, linewidth=1, edgecolor="r", facecolor="none")
        # Dibujar el rectangulo
        #ax[1].add_patch(rect)

        # Dibujar cada linea
        (x0, y0) = dist * np.array([np.cos(theta), np.sin(theta)])
        ax[2].axline((x0, y0), slope=np.tan(theta + np.pi/2), color="r")
    
    ax[2].axis((0, col1, row1, 0))
    ax[2].set_title("Lineas detectadas")
    ax[2].set_axis_off()

# Crear Imagen de muestra
image = np.zeros((100, 100))
idx = np.arange(25, 75)
image[idx[::-1], idx] = 255
image[idx, idx] = 255

# Probar la transformada con la imagen muestra (imagen de una X)
transformadaHough(image)

atico = cv2.imread("Practica8_Hough_GomezCanales/atico.jpg", cv2.IMREAD_GRAYSCALE)
# Binarizar imagen
atico_b = cv2.Canny(atico, 50, 200, None, 3)

plt.figure()
plt.imshow(atico, cmap="gray")
plt.title("Imagen Original Ático")

transformadaHough(atico_b)
plt.show()