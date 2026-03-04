import os
import csv
from pathlib import Path

def create_labels_csv(base_dir, output_csv):
    base_path = Path(base_dir)
    negative_dir = base_path / 'negative_IDC'
    positive_dir = base_path / 'positive_IDC'
    
    # Contadores para mostrar un resumen al final
    count_negative = 0
    count_positive = 0
    
    print(f"Buscando imágenes en {base_dir}...")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'label'])
        
        # Procesar negative_IDC (label 0)
        if negative_dir.exists():
            for filepath in negative_dir.rglob('*.*'):
                if filepath.is_file() and not filepath.name.startswith('.'):
                    # Usamos la ruta relativa para el archivo CSV (ej: negative_IDC/imagen.png)
                    rel_path = filepath.relative_to(base_path)
                    writer.writerow([str(rel_path), 0])
                    count_negative += 1
        else:
            print(f"Advertencia: No se encontró la carpeta {negative_dir}")
                    
        # Procesar positive_IDC (label 1)
        if positive_dir.exists():
            for filepath in positive_dir.rglob('*.*'):
                if filepath.is_file() and not filepath.name.startswith('.'):
                    rel_path = filepath.relative_to(base_path)
                    writer.writerow([str(rel_path), 1])
                    count_positive += 1
        else:
            print(f"Advertencia: No se encontró la carpeta {positive_dir}")

    print(f"¡Hecho! Se procesaron {count_negative} imágenes negativas (0) y {count_positive} positivas (1).")
    print(f"El archivo CSV se guardó en: {output_csv}")

if __name__ == '__main__':
    # Rutas definidas según tu solicitud
    base_directory = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5'
    output_csv_file = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5/labels.csv'
    
    create_labels_csv(base_directory, output_csv_file)
