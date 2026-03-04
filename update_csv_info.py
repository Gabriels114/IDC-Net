import csv
import os
import re

input_csv = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5/labels_shuffled.csv'
output_csv = '/Users/gabriels/Proyectos/Samsung/Dataset1/IDC_regular_ps50_idx5/labels_shuffled_updated.csv'

pattern = re.compile(r'^(.+)_x(\d+)_y(\d+)_class(\d+)\.png$')

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['patient_id', 'x', 'y', 'class']
    
    # Avoid duplicate columns if run multiple times
    seen = set()
    unique_fieldnames = []
    for f in fieldnames:
        if f not in seen:
            unique_fieldnames.append(f)
            seen.add(f)
            
    writer = csv.DictWriter(outfile, fieldnames=unique_fieldnames)
    writer.writeheader()
    
    for row in reader:
        filename = row.get('filename', '')
        basename = os.path.basename(filename)
        match = pattern.match(basename)
        if match:
            # Add or update keys
            row['patient_id'] = match.group(1)
            row['x'] = match.group(2)
            row['y'] = match.group(3)
            row['class'] = match.group(4)
        else:
            row['patient_id'] = ''
            row['x'] = ''
            row['y'] = ''
            row['class'] = ''
            
        writer.writerow(row)

os.replace(output_csv, input_csv)
print("CSV updated successfully.")
