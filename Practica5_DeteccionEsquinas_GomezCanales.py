# Angel Isaac Gomez Canales
# Deteccion de Esquinas usando el metodo de Harris

# Importar librerias necesarias
import numpy as np
import cv2 as cv

def filtrar_box(src, n):
    """
    Aplica un filtro box de nxn
    - input: 
        - src: imagen que se desea filtrar
        - n: tamano del filtro (nxn)
    - output: imagen filtrada
    """
    h_box = np.ones((n, n), dtype=np.float32) / (n * n)
    src_filt = cv.filter2D(src, -1, h_box)
    return src_filt

def derivadas(src):
    """
    Calcular las derivadas en x y en y de la imagen usando el kernel de sobel
    - input: imagen a calcular derivadas
    - output: Ix, Iy, derivadas en x y en y, respectivamente
    """
    # Kernels sobel en x y en y
    hx = np.array([[-1, 0, 1], 
                   [-2, 0, 2],
                   [-1, 0, 1]])
    hy = np.array([[-1, -2, -1], 
                   [0, 0, 0],
                   [1, 2, 1]])

    Ix = cv.filter2D(src, -1, hx)
    Iy = cv.filter2D(src, -1, hy)

    return Ix, Iy

def momentos(Ix, Iy):
    """
    Obtener los momentos (A, B y C)
    - input: 
        - Ix: Derivada de la imagen en x
        - Iy: Derivada de la imagen en y
    - output: momentos A, B y C
    """
    M11 = Ix * Ix
    M22 = Iy * Iy
    M12 = Ix * Iy

    # Kernel Gaussiano
    hg = np.matrix('1 4 6 4 1;4 16 24 16 4;6 24 36 24 6;4 16 24 16 4;1 4 6 4 1')
    hg = hg * (1/256)

    A = cv.filter2D(M11, -1, hg)
    B = cv.filter2D(M22, -1, hg)
    C = cv.filter2D(M12, -1, hg)
    return A, B, C

def NMS(U, V, vecindad):
    """
    Non Maximum Supresion para obtener esquinas
    - input:
        - U: Imagen binaria con esquinas detectadas
        - V: Corner Response
        - vecindad: tamano de la ventana
    - output: imagen con esquinas detectadas con NMS
    """
    mask = np.zeros_like(U)
    M, N = U.shape
    # Iterar sobre cada pixel de la imagen
    for r in range(M):
        for c in range(N):
            if U[r, c]:  # Si el pixel es 1
                # Ajustar la ventana para que no este fuera de los limites de la imagen
                I1 = np.array([r - vecindad, 0])
                I2 = np.array([r + vecindad, M])
                I3 = np.array([c - vecindad, 0])
                I4 = np.array([c + vecindad, N])
                datxi = np.max(I1)
                datxs = np.min(I2)
                datyi = np.max(I3)
                datys = np.min(I4)

                maxB = np.max(V[datxi:datxs, datyi:datys])  # Obtener el maximo dentro de la ventana

                if V[r, c]  == maxB:
                    mask[r, c] = 1
    return mask

def dibujar_esquinas(src, mask):
    """
    Dibujar las esquinas sobre una imagen a color
    - input: 
        - src: imagen a color sobre la que se dibujaran las esquinas
        - mask: mascara con la ubicacion de las esquinas
    """
    M, N = mask.shape
    for r in range(M):
        for c in range(N):
            if mask[r, c]:  # Si el pixel es esquina(1)
                # cv2.drawMarker(img, (x, y), color, markerType, markerSize, thickness)
                cv.drawMarker(src, (c, r), (0, 0, 255), cv.MARKER_CROSS)

# Configurar webcam
cap = cv.VideoCapture(0)
# Ajustar tamano de la imagen
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

#Trackbar para el umbral y vecinos
def th(x):
    pass

def vecindad(x):
    pass

# Crear ventana
cv.namedWindow("Deteccion de esquinas")
# Crear Trackbar
cv.createTrackbar("th", "Deteccion de esquinas", 1, 20, th)
cv.createTrackbar("vecindad", "Deteccion de esquinas", 1, 100, vecindad)

alpha = 0.01

while True:
    # Obtener imagen de la webcam
    _, src = cap.read()
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # Convertir imagen a escala de grises
    src_gray = src_gray.astype(np.float64)  # Convertir a floats para poder operarla

    # Suavizar la imagen
    src_gray = filtrar_box(src_gray, 5)

    # Calcular derivadas
    Ix, Iy = derivadas(src_gray)

    # Obtener momentos
    A, B, C = momentos(Ix, Iy)

    # Corner Response
    V = (A * B) - (C**2) - alpha * ((A + B) * (A + B))
    th = cv.getTrackbarPos("th", "Deteccion de esquinas") * 1e7
    vecindad = cv.getTrackbarPos("vecindad", "Deteccion de esquinas") 
    U = V > th  # Umbralizar
    # 4 y 21
    # Realizar Non Maximum Supresion para qudarnos con esquinas unicas
    S = NMS(U, V, vecindad)

    # Dibujar las esquinas y mostrar la imagen
    dibujar_esquinas(src, S)
    cv.imshow("Deteccion de esquinas", src)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break