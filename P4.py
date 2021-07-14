''' Proyecto 4
Modelos probabilisticos de senales y sistemas
Diego Valladares Porras
B77867
'''

# Para visualizar imágenes y señales
import matplotlib.pyplot as plt

# Para medir el tiempo de simulación
import time

from PIL import Image
import numpy as np

from scipy import fft

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

# 4.1. - Modulación 16-QAM

def modulador_QAM(bits, fc, mpp):
    '''Un método que simula el esquema de
    modulación digital QAM.
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    # Ambas portadoras de la señal:
    portadora_I = np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    # Se definen las componentes en I y Q
    senal_Tx_I = np.zeros(t_simulacion.shape)
    senal_Tx_Q = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # señal de información
     
    j = 0
    
    # 4. Asignar las formas de onda según los bits (QAM)
    for i in range(0,N,4):
    
        if i < N+4: # Para evitar sobrepaso

        # Portadora I
        
            if bits[i] == 0 and bits[i+1] == 0:
                senal_Tx_I[j*mpp : (j+1)*mpp] = portadora_I*(-3)
            elif bits[i] == 0 and bits[i+1] == 1:
                senal_Tx_I[j*mpp : (j+1)*mpp] = portadora_I*(-1)
            elif bits[i] == 1 and bits[i+1] == 0:
                senal_Tx_I[j*mpp : (j+1)*mpp] = portadora_I*3
            elif bits[i] == 1 and bits[i+1] == 1:
                senal_Tx_I[j*mpp : (j+1)*mpp] = portadora_I*(1)
                        
        # Portadora Q
        
            if bits[i+2] == 0 and bits[i+3] == 0:
                senal_Tx_Q[j*mpp : (j+1)*mpp] = portadora_Q*(3)
            elif bits[i+2] == 0 and bits[i+3] == 1:
                senal_Tx_Q[j*mpp : (j+1)*mpp] = portadora_Q*1
            elif bits[i+2] == 1 and bits[i+3] == 0:
                senal_Tx_Q[j*mpp : (j+1)*mpp] = portadora_Q*(-3)
            elif bits[i+2] == 1 and bits[i+3] == 1:
                senal_Tx_Q[j*mpp : (j+1)*mpp] = portadora_Q*(-1)
            j = j+1
         
    # 5. Calcular la potencia promedio de la señal modulada
    # Se suman las componentes en I y en Q
    senal_Tx = senal_Tx_I + senal_Tx_Q
    
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, Pm, portadora_I, portadora_Q, moduladora 

''' Ahora se redefine la función 'demulador' para adaptarla a los cambios anteriores. '''

def demodulador_QAM(senal_Rx, portadora_I, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema !QAM El criterio
    de demodulación se basa en decodificación por 
    detección de energía.
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

     # Demodulación
    
    j =0
    for i in range(N):
        
        if j+4 > N:
            break
            
        # Producto interno de dos funciones
        producto_I = senal_Rx[i*mpp : (i+1)*mpp] * portadora_I
        producto_Q = senal_Rx[i*mpp : (i+1)*mpp] * portadora_Q
        senal_demodulada[i*mpp : (i+1)*mpp] = producto_I + producto_Q

        # Criterio de decisión por detección de energía
        # Originalmente se podía comparar Ep > 0, pero para tomar en cuenta ambos 
        # componentes, se compara con la suma de todos los casos y se usa un 1 en vez de 0
        if np.sum(producto_I) >= 0:
            bits_Rx[j] = 1  # b1

        if np.max(np.abs(producto_I)) < 2.5:
           bits_Rx[j+1] = 1  # b2

        if np.sum(producto_Q) < 0:
            bits_Rx[j+2] = 1  # b3

        if np.max(np.abs(producto_Q)) < 2.5:
            bits_Rx[j+3] = 1  # b4
        j += 4

    return bits_Rx.astype(int), senal_demodulada



def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


''' ***** Ahora que se redefinieron las funciones necesarias, se sigue el mismo procedimiento del ejemplo, pero utilizando 
la función modificadas. ***** '''

# Primero se grafican la imagen enviada y la recuperada

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 50    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema QAM
senal_Tx, Pm, portadora_I, portadora_Q, moduladora = modulador_QAM(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador_QAM(senal_Rx, portadora_I, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)


# Ahora se muestran las señales

# Visualizar el cambio entre las señales
fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
#ax1.plot(moduladora[0:600], color='r', lw=2) 
#ax1.set_ylabel('$b(t)$')

# La señal modulada por QAM
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()


#4.2

'''Para evaluar la estacionaridad, se calcula el promedio de la amplitud. La señal tx está formada por cuatro componentes:
cos+sin, cos-sin, -cos+sin, -cos-sin. Primero se obtiene la amplitud de cada uno:'''

cosMasSin = [np.cos(x) + np.sin(x) for x in range(10)]
cosMenosSin = [np.cos(x) - np.sin(x) for x in range(10)]
menosCosMenosSin = [- np.cos(x) - np.sin(x) for x in range(10)]
menosCosMasSin = [- np.cos(x) + np.sin(x) for x in range(10)]

amplitud1 = np.max(cosMasSin)
amplitud2 = np.max(cosMenosSin)
amplitud3 = np.max(menosCosMasSin)
amplitud4 = np.max(menosCosMenosSin)

promedio = (amplitud1 + amplitud2 + amplitud3 + amplitud4)/4

print("El promedio de la amplitud es ", promedio)
print("Ese valor está cercano a 1, que es cercano a lo esperado.")

print()
print()
'''Para evaluar la ergodicidad, se obtiene el promedio en el tiempo y el estadístico, y se comparan.'''

senal_Tx_1 = senal_Tx[0:20] # Un periodo de la señal, ya que hay 20 muestras por periodo

print("Promedio estadístico: ", np.mean(senal_Tx))
print("Promedio en el tiempo: ", np.mean(senal_Tx_1))
print("Los valores son medianamente similares, por lo que la ergodicidad no es total.")



#4.3

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 12000)
plt.grid()
plt.title('Densidad espectral de potencia y componentes de la moduladora')
plt.xlabel('Frecuencia')
plt.ylabel('Densidad espectral de potencia')
plt.show()