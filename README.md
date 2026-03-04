# 🧬 IDC-Net  
### *Deep Learning para la Detección de Carcinoma Ductal Invasivo*

---

## 📌 Descripción General

**IDC-Net** es un modelo de *Deep Learning* diseñado para la detección automática de **Carcinoma Ductal Invasivo (IDC)** en imágenes histopatológicas de mama.

El sistema opera a nivel de **parches (50×50 píxeles)** y realiza una clasificación binaria entre:

- **IDC(-)** — Tejido no invasivo  
- **IDC(+)** — Presencia de carcinoma ductal invasivo  

Este proyecto contribuye al desarrollo de sistemas de apoyo al diagnóstico (CAD) mediante modelos escalables, reproducibles y clínicamente relevantes en patología digital.

---

## 🧠 Planteamiento del Problema

El Carcinoma Ductal Invasivo (IDC) es el subtipo más común de cáncer de mama.  
Para determinar el grado de agresividad tumoral, los patólogos analizan regiones específicas del tejido que contienen IDC.

La identificación automática de estas regiones:

- Reduce la carga diagnóstica  
- Mejora la reproducibilidad clínica  
- Permite análisis a gran escala  
- Facilita el desarrollo de modelos posteriores de gradación tumoral  

IDC-Net se enfoca en la clasificación a nivel de parche como paso previo a la delimitación completa en imágenes de lámina completa (*Whole Slide Images*).

---

## 📂 Conjunto de Datos

**Dataset:** Breast Histopathology Images (IDC Regular Patches)

- 162 imágenes de lámina completa escaneadas a 40x  
- 277,524 parches extraídos de tamaño 50×50 píxeles  
- Subconjunto balanceado utilizado:
  - 78,786 IDC(-)
  - 78,786 IDC(+)

