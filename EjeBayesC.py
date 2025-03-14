import numpy as np
import pandas as pd
from scipy.stats import norm

# Ejercicio 1: Clasificación de frutas (Una característica)
data_fruits = pd.DataFrame({
    "Fruta": ["Manzana", "Pera"],
    "Probabilidad": [0.6, 0.4],
    "Roja": [0.7, 0.2]
})

# Calcular la probabilidad total de que una fruta sea roja
P_R = (data_fruits["Probabilidad"] * data_fruits["Roja"]).sum()

# Calcular la probabilidad de que sea una manzana dado que es roja
P_M_R = (data_fruits.loc[data_fruits["Fruta"] == "Manzana", "Roja"].values[0] * 
         data_fruits.loc[data_fruits["Fruta"] == "Manzana", "Probabilidad"].values[0]) / P_R

print(f"Ejercicio 1: La probabilidad de que la fruta roja sea una manzana es: {P_M_R:.2%}")

# Ejercicio 2: Detección de correo spam (Tres características)
data_email = pd.DataFrame({
    "Tipo": ["Spam", "Normal"],
    "Probabilidad": [0.3, 0.7],
    "Oferta": [0.9, 0.2],
    "Adjunto": [0.8, 0.4],
    "Link": [0.85, 0.1]
})

# Calcular la probabilidad de que el correo tenga las 3 características
data_email["P_C"] = data_email["Oferta"] * data_email["Adjunto"] * data_email["Link"]
P_C = (data_email["Probabilidad"] * data_email["P_C"]).sum()

# Calcular la probabilidad de que sea spam dado que tiene las 3 características
P_S_C = (data_email.loc[data_email["Tipo"] == "Spam", "P_C"].values[0] * 
         data_email.loc[data_email["Tipo"] == "Spam", "Probabilidad"].values[0]) / P_C

print(f"Ejercicio 2: La probabilidad de que el correo sea spam es: {P_S_C:.2%}")

# Extra: Uso de norm (Distribución Normal)
# Ejemplo de una distribución normal con media 0 y desviación estándar 1
x = np.linspace(-3, 3, 100)
pdf_values = norm.pdf(x, loc=0, scale=1)  # Función de densidad de probabilidad

# Crear un DataFrame con la distribución normal
df_norm = pd.DataFrame({"x": x, "pdf": pdf_values})

# Mostrar las primeras filas
print("\nEjemplo de distribución normal:\n", df_norm.head())
