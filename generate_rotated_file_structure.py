import os
import shutil

# Pfad zum Ordner mit den Bildern
image_folder = "val2017_rotated"

# Zielordner für die sortierten Bilder erstellen
sorted_folder = "val2017_rotated_subfolder"
os.makedirs(sorted_folder, exist_ok=True)

# Alle Bilder im Ordner auflisten
image_files = os.listdir(image_folder)

# Bilder in separate Unterordner basierend auf ihren Winkeln verschieben
for image_file in image_files:
    # Den Winkel aus dem Dateinamen extrahieren
    angle = image_file.split("_")[2]

    # Den Pfad zum aktuellen Bild erstellen
    image_path = os.path.join(image_folder, image_file)

    # Den Pfad zum Zielordner für den aktuellen Winkel erstellen
    angle_folder = os.path.join(sorted_folder, angle)
    os.makedirs(angle_folder, exist_ok=True)

    # Das Bild in den entsprechenden Unterordner verschieben
    shutil.move(image_path, os.path.join(angle_folder, image_file))