# Angel Isaac Gomez Canales
# Deteccion de Imagenes Borrosas

# Importar librerias necesarias
import numpy as np
import cv2 as cv

# Funcion para genera filtro gaussiano en el dominio de la frecuencia
def generar_LPGaussiano(sigma, size: tuple):
    """
    Esta funcion genera el kernel de un filtro pasa bajas gaussiano en dominio de frecuencia
    input: 
        sigma: desviacion estandar
        size: tamano del filtro
    output: filtro gaussiano en dominio de frecuencia
    """
    M, N = size
    # Obtener tamano del filtro
    if M % 2 == 0:
        r = np.arange(-np.floor(M/2), np.floor(M/2), 1)
    else: 
        r = np.arange(np.arange(-np.floor(M/2), np.floor(M/2)+1, 1))
    if N % 2 == 0:
        c = np.arange(-np.floor(N/2), np.floor(N/2), 1)
    else: 
        c = np.arange(np.arange(-np.floor(N/2), np.floor(N/2)+1, 1))
    C, R = np.meshgrid(c, r)
    D2 = C ** 2 + R ** 2
    H_LP_G = np.exp(-D2 / (2 * (sigma**2)))
    # Hacer el inverse shift para ponerlo en la posicion correcta
    return np.fft.ifftshift(H_LP_G)

def aplicarFiltFrec(Im, H):
    """
    Esta funcion aplica filtros en frecuencia
    input: 
        Im: Imagen que se desea filtrar
        H: Filtro en frecuenca
    output: imagen filtrada
    """
    FIm = np.fft.fft2(Im)
    ImFilt = FIm * H

# Capturar video
cap = cv.VideoCapture(0)
# Ajustar tamano de la imagen
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

# Definir trackbar para tamano filtro gaussiano y su desvesta
def n(x):
    pass

def sigma(x):
    pass

# Crear ventanas
cv.namedWindow("Imagen Filtrada")
# Crear trackbar
cv.createTrackbar("sigma", "Imagen Filtrada", 1, 200, sigma)

while True:
    # Obtener imagen de la webcam
    _, src = cap.read()
    # Convertir a escala de grises
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("Imagen escala de grises", src)

    # Filtrar la imagen en frecuencia ---------------------
    # Calcular transformada de fourier da la imagen
    srcfft = np.fft.fft2(src)
    # Obtener sigma del trackbar
    sigma = cv.getTrackbarPos("sigma", "Imagen Filtrada")
    # Generar filtro
    H = generar_LPGaussiano(sigma, src.shape)
    # Mostrar el filtro
    cv.imshow("Filtro gaussiano", (np.fft.fftshift(H) * 255).astype(np.uint8))
    # Filtrar imgagen
    src_filt = np.real(np.fft.ifft2(srcfft * H)).astype(np.uint8)

    # Obtner la magnitud de la transformada de la imagen filtrada
    Fsrc_filt_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(src_filt))))
    # Media de la magnitud
    media = np.mean(Fsrc_filt_mag)
    # Es borrosa si la media es menor a 6
    borrosa = media < 5.65

    # EScribir la media y el resultado sobre la imagen
    img = cv.UMat(np.dstack([src_filt] * 3))
    color = (0, 0, 255) if borrosa else (0, 255, 0)
    texto = f"Media: {media:0.2f}   Imagen Borrosa" if borrosa else f"Media: {media:0.2f}   Imagen Nitida"
    cv.putText(img, texto, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostrar imagen filtrada
    cv.imshow("Imagen Filtrada", src_filt)

    # Mostrar Resultado
    cv.imshow("Deteccion de Imagne Borrosa", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


