import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
BASE_DIR = '/Users/gabriels/Proyectos/Samsung/Imagenes'
CSV_PATH = os.path.join(BASE_DIR, 'labels.csv')
IMG_SIZE = (50, 50)
BATCH_SIZE = 64
EPOCHS = 20

def load_and_split_data(csv_path, test_size=0.15, val_size=0.15):
    """
    Carga el CSV y divide los datos asegurando que todos los parches de un 
    mismo paciente (patient_id) estén estrictamente en uno de los conjuntos.
    """
    print(f"Cargando dataset desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Asegurarnos de que las clases son strings para ImageDataGenerator
    df['class'] = df['class'].astype(str)
    
    print(f"Total de imágenes: {len(df)}")
    print(f"Pacientes únicos: {df['patient_id'].nunique()}")

    # 1. Separar un conjunto de PRUEBA (Test)
    # GroupShuffleSplit asegura que los grupos (pacientes) no se mezclen.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['patient_id']))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 2. Separar el restante en ENTRENAMIENTO y VALIDACIÓN
    # Calculamos la proporción de validación relativa al set restante
    # Si test_size es 0.15 y val_size es 0.15, val es ~0.176 del 0.85 restante
    val_relative_size = val_size / (1.0 - test_size)
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['patient_id']))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    print("\n--- Distribución de splits por paciente (patient_id) ---")
    print(f"Train: {len(df_train)} imgs (pacientes: {df_train['patient_id'].nunique()})")
    print(f"Val:   {len(df_val)} imgs (pacientes: {df_val['patient_id'].nunique()})")
    print(f"Test:  {len(df_test)} imgs (pacientes: {df_test['patient_id'].nunique()})")
    
    # 3. Validar Fuga de Datos (Data Leakage)
    train_patients = set(df_train['patient_id'])
    val_patients = set(df_val['patient_id'])
    test_patients = set(df_test['patient_id'])
    
    assert len(train_patients.intersection(test_patients)) == 0, "¡FUGA DE DATOS entre Train y Test!"
    assert len(train_patients.intersection(val_patients)) == 0, "¡FUGA DE DATOS entre Train y Val!"
    print("Verificación de fuga de datos (Data Leakage) pasada: OK.\n")

    return df_train, df_val, df_test

def create_data_generators(df_train, df_val, df_test, base_dir):
    """Crea los generadores de imágenes (Image Data Generators) con Data Augmentation básico para predecir."""
    
    # Data Augmentation básico para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Solo reescalado para validación y prueba
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Preparando generador de Entrenamiento...")
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=base_dir,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    print("Preparando generador de Validación...")
    val_generator = test_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=base_dir,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    print("Preparando generador de Prueba...")
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=base_dir,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False # IMPORTANTE: no mezclar test para comparar etiquetas
    )

    return train_generator, val_generator, test_generator

def build_model():
    """Construye y compila el modelo CNN"""
    model = Sequential([
        # Bloque 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloque 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloque 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Clasificador Fully Connected
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Clasificación Binaria
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def plot_metrics(history):
    """Grafica la precisión y pérdida durante el entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Gráfica de historial guardada en 'training_history.png'")
    # plt.show() # Descomentar para ver en notebook o entorno visual

def evaluate_model(model, test_generator):
    """Evalúa el modelo y genera el reporte de clasificación y matriz de confusión."""
    print("\nEvaluando el modelo en el conjunto de prueba...")
    
    # Calcular pasos requeridos para pasar todo el set de test exactamente una vez
    test_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    
    # Obtener probabilidades
    predictions = model.predict(test_generator, steps=test_steps)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    
    # Etiquetas reales
    y_true =  test_generator.classes
    
    # Nombres de las clases
    class_labels = list(test_generator.class_indices.keys())
    
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true, y_pred, target_names=["Negativo (0)", "Positivo (1)"]))
    
    # Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Predicción Negativo", "Predicción Positivo"],
                yticklabels=["Real Negativo", "Real Positivo"])
    plt.title('Matriz de Confusión (Test)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Gráfica de matriz de confusión guardada en 'confusion_matrix.png'")
    

if __name__ == "__main__":
    # 1. Cargar y dividir
    df_train, df_val, df_test = load_and_split_data(CSV_PATH)
    
    # 2. Generar dataloaders
    train_gen, val_gen, test_gen = create_data_generators(df_train, df_val, df_test, BASE_DIR)
    
    # 3. Construir modelo
    model = build_model()
    model.summary()
    
    # 4. Callbacks de entrenamiento
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    
    # Cálculo de pesos para balancear la clase si es necesario (el dataset original es balanceado, pero el split puede variar levemente)
    neg_count = sum(df_train['class'] == '0')
    pos_count = sum(df_train['class'] == '1')
    total = len(df_train)
    weight_0 = (1 / neg_count) * (total / 2.0)
    weight_1 = (1 / pos_count) * (total / 2.0)
    class_weights = {0: weight_0, 1: weight_1}
    print(f"\nPesos de clases: 0: {weight_0:.3f}, 1: {weight_1:.3f}\n")

    # 5. Entrenar (Configurado EPOCHS a 1 solo para probar rápido. Cambiar a 20 o más después)
    # Por defecto está a EPOCHS (20) en las constantes arriba.
    print("Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stop]
    )
    
    # 6. Graficar y Evaluar
    plot_metrics(history)
    evaluate_model(model, test_gen)
    
    # 7. Guardar el modelo
    model.save('idc_breast_cancer_model.h5')
    print("Modelo guardado exitosamente como 'idc_breast_cancer_model.h5'")
