import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities


###   Hilfsfunktionen  ###
# Randbehandlung:

### Randbehandlung ###
# 🔹 Randbehandlung-Funktion
def applyBorderHandling(img, border_type):
    """Wendet eine Randbehandlung auf das Bild an, ohne die Größe zu verändern."""
    if border_type == "Extrapolieren":
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    elif border_type == "Spiegeln":
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    elif border_type == "Zyklisch":
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_WRAP)
    elif border_type == "Nullen":
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    else:
        print("⚠️ Ungültige Randbehandlung. Standardmäßig: Spiegeln.")
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)

##########################################################################################
# Filter funktionenunter anwednugn der falterung im raum :
##########################################################################################
######################################################################

# apply median filter
# Median Filter
    # Grauwerte der Nachbarschaft sortieren und zuordnen
def applyMedianFilter(img, kSize, borderType="spiegeln"):
 #   img = applyNachbarschaftsoperator(img, borderType)
 #   MD_filtered = cv2.medianBlur(img, kSize)
 #   print(f"Median Filter mit kSize={kSize} angewendet")
  #  return MD_filtered
    print("Applying Median Filter")
    pass
######################################################################

# Moving Average Filter
    # Kernel definieren

    # Faltung im Bildraum
# create a moving average kernel of arbitrary size
def createMovingAverageKernel(kSize):
 #   """Erzeugt einen Moving Average Kernel der Größe kSize x kSize"""
 #   kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
 #   print(f"Moving Average Kernel mit kSize={kSize} erstellt:\n{kernel}")
 #   return kernel
 print("Moving Average Kernel erstellt")
 pass

def applyMovingAverageFilter(img, kSize, borderType):
 #   kernel = createMovingAverageKernel(kSize)
#    MVA_filtered = cv2.filter2D(img, -1, kernel, borderType=borderType)
  #  print(f"Applying Moving Average Filter mit kSize={kSize}, borderType={borderType}")
  #  return MVA_filtered
   print("Applying Moving Average Filter")
   pass

##########################################################################

# Gaussian Filter
    # Kernel definieren
    # Faltung im Bildraum
def createGaussianKernel(kSize, sigma=1.0):
    """Erzeugt einen Gauß-Kernel der Größe kSize x kSize mit Standardabweichung sigma"""
    print(f"Generiere Gauß-Kernel mit kSize={kSize}, sigma={sigma}")
    ax = np.linspace(-(kSize // 2), kSize // 2, kSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalisierung
    print("Gauß-Kernel erzeugt:\n", kernel)
    return kernel



def applyGaussianFilter(img, kSize=5, sigma=1.0,border_type="Spiegeln"):
    """Wendet einen Gauß-Filter mit der gewählten Randbehandlung an."""
    print(f"Wende Gaußfilter an mit Randbehandlung: {border_type}")

    # Randbehandlung anwenden
    img_with_border = applyBorderHandling(img, border_type)

    # Kernel erstellen
    kernel = createGaussianKernel(kSize, sigma)

    # Faltung durchführen
    filtered_img = cv2.filter2D(img_with_border, -1, kernel, borderType=cv2.BORDER_DEFAULT)

    return filtered_img

# 🔹 Test-Code zum Laden & Anwenden des Filters
if __name__ == "__main__":
    img = cv2.imread("testbild.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("⚠️ Fehler: Bild konnte nicht geladen werden!")
    else:
        print("✅ Bild erfolgreich geladen.")

        # Randbehandlung auswählen
        border_type = "Spiegeln"  # Alternativen: "Extrapolieren", "Zyklisch", "Nullen"

        # Filter anwenden
        gauss_filtered = applyGaussianFilter(img, kSize=5, sigma=1.0, border_type=border_type)

        # Bilder anzeigen
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Originalbild")
        plt.imshow(img, cmap='gray')

        plt.subplot(1,2,2)
        plt.title(f"Gauß-gefiltertes Bild ({border_type})")
        plt.imshow(gauss_filtered, cmap='gray')

        plt.show()

        # Speichern
        cv2.imwrite("gauss_filtered_border.png", gauss_filtered)
        print("✅ Gefiltertes Bild gespeichert als 'gauss_filtered_border.png'.")

######################################################################

# Sobel Filter
    # Kernel definieren
    # Faltung im Bildraum

# create a sobel kernel in x direction of size 3x3
def createSobelXKernel():
  #  """Erzeugt den Sobel X Kernel"""
   # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    #print("Sobel X Kernel erstellt:\n", kernel)
   # return kernel
    print("Sobel X Kernel erstellt")
    pass
# create a sobel kernel in y direction of size 3x3
def createSobelYKernel():
  #  """Erzeugt den Sobel Y Kernel"""
  #  kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
  #  print("Sobel Y Kernel erstellt:\n", kernel)
   # return kernel
    print("Sobel Y Kernel erstellt")
    pass

def applySobelFilter(img, direction="x", borderType=cv2.BORDER_REFLECT):
  #  print(f"Applying Sobel Filter in {direction}-Richtung mit borderType={borderType}")
   # kernel = createSobelXKernel() if direction == "x" else createSobelYKernel()
   # return cv2.filter2D(img, -1, kernel, borderType=borderType)
    print(f"Applying Sobel Filter in {direction}-Richtung")
    pass
######################################################################

def applyKernelInSpatialDomain(img, kernel, borderType=cv2.BORDER_REFLECT):
    # print(f"Applying kernel in spatial domain mit borderType={borderType}")
    # return cv2.filter2D(img, -1, kernel, borderType=borderType)
    print ("Applying kernel in spatial domain")
    pass



######################################################################
####################################################################
# Extra: create an integral image of the given image
def createIntegralImage(img):
    print("Integralbild erstellt")
    integral_image = img.copy()
    return integral_image


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    print(f"Applying Moving Average Filter mit Integralbild, kSize={kSize}")
    filtered_img = img.copy()
    return filtered_img


# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    print(f"Applying Moving Average Filter mit separierten Kernen, kSize={kSize}")

    filtered_img = img.copy()
    return filtered_img

def run_runtime_evaluation(img):
    print("Runtime Evaluation gestartet")
    pass