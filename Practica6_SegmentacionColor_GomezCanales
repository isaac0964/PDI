# Angel Isaac Gomez Canales
# Segmentacion por color 07/03/2024

# Importar librerias
import cv2 as cv
import numpy as np

# Funcion para mostrar video y seleccionar ROI
def seleccionar_ROI(cap):
    """
    Esta funcion despliega el video y espera por 'q' para hacer la captura
    con el mouse seleccionar la ROI y tecla enter
    - Input: cap: webcam
    - Output: crop: ROI recortada de la imagen
    """
    while True:
        # Mostrar webcam
        _, I = cap.read()
        cv.imshow("Imagen (presionar q para congelar img)", I)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    # Seleccionar roi
    roi = cv.selectROI("Seleccione la ROI y presione enter", I)
    # Recortar la ROI de la imagen
    roi_cropped = I[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv.destroyAllWindows()
    return roi_cropped

def segmentacionColor(I, roi, ns):
    """
    Esta funcion segmenta un cierto color dentro de una imagen
    - Input: 
        - I: Imagen sobre la que se desea hacer la segmentacion
        - roi: Region de interes que contiene el color a segmentar
        - ns: numero de desviaciones estandar de tolerancia par segementar
    - Output:
        - I_seg: Imagen con el color detectado encerrado dentro de un rectangulo
        - I_bw: Imagen blanco y negro con los pixeles correspondientes al color pintados de blanco y el resto en negro
    """
    # Obtener la media y desviacion estandar de cada canal en la roi
    b, g, r = cv.split(roi)
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    
    # Obtener posiciones donde la intensidad de los 3 canales cumple con:
    # mean - ns * std < I(x, y) < mean + ns * std
    pos_filas, pos_cols = np.where((((mean_r - ns*std_r < I[:, :, 2]) & (I[:, :, 2] < mean_r + ns*std_r)) &
                    ((mean_g - ns*std_g < I[:, :, 1]) & (I[:, :, 1]< mean_g + ns*std_g)) &
                    ((mean_b - ns*std_b < I[:, :, 0]) & (I[:, :, 0] < mean_b + ns*std_b))))
    
    # Generar imagen blanco y negro con los resultados de la segmentacion
    I_bw = np.zeros((I.shape[:-1]), dtype=np.uint8)
    I_bw[pos_filas, pos_cols] = 255

    # Encerrar los objetos
    try: 
        bf, bc = np.where(I_bw)  # Coords de los pixeles en blanco
        min_fila = np.min(bf)
        min_col = np.min(bc)
        max_fila = np.max(bf)
        max_col = np.max(bc)
        # Inicio y final del rectangulo para encerrar objeto
        start = [min_col, min_fila]
        end = [max_col, max_fila]
        # Dibujar rectangulo
        I = cv.rectangle(I, start, end, (0, 0, 255), 2)
        # Encontrar centro del rectangulo
        # image = cv2.rectangle(image, start_point, end_point, color, thickness)
        centerx = min_col + (max_col - min_col) // 2
        centery = min_fila + (max_fila - min_fila) // 2
        # Dibujar circulo
        # image = cv.circle(image, centerOfCircle, radius, color, thickness)
        I = cv.circle(I, [centerx, centery], 5, (0, 0, 255), -1)
    except:
        # En caso de que no se detecte ningun objeto del color deseado
        print("No se ha encontrado ningun objeto con el color deseado")
        return
    return I, I_bw

def ns(x):
    pass

# Configurar webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Seleccionar ROI
roi = seleccionar_ROI(cap)

# Crear trackbar y ventana
cv.namedWindow("Imagen segmentada")
cv.createTrackbar("ns", "Imagen segmentada", 2, 5, ns)
# Detectar el color en la webcam
while True:
    # Leer imagen de la webcam
    _, I = cap.read()
    # Obtener el numero de desviaciones estandar del trackbar
    ns = cv.getTrackbarPos("ns", "Imagen segmentada")
    # Segmentar la imagen
    I, I_bw = segmentacionColor(I, roi, ns)
    cv.imshow("Imagen segmentada", I)
    cv.imshow("Imagen blanco y negro de segmentacion", I_bw)
    if cv.waitKey(1) == ord('q'):
        break