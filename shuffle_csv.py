import csv
import random
import sys

def shuffle_csv(input_csv, output_csv):
    print(f"Leyendo el archivo: {input_csv}...")
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Extraer el encabezado (primera fila)
            header = next(reader)
            # Leer el resto de las filas
            rows = list(reader)
            
        print(f"Se leyeron {len(rows)} filas.")
        print("Mezclando de forma aleatoria (shuffling)...")
        # Mezclar las filas
        random.seed(42)  # Semilla fija para reproducibilidad
        random.shuffle(rows)
        
        print(f"Escribiendo resultado en: {output_csv}...")
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Escribir el encabezado de nuevo
            writer.writerow(header)
            # Escribir los datos ya mezclados
            writer.writerows(rows)
            
        print("¡Proceso completado exitosamente!")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        sys.exit(1)

if __name__ == '__main__':
    input_file = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5/labels.csv'
    output_file = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5/labels_shuffled.csv'
    shuffle_csv(input_file, output_file)
