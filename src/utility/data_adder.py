import csv
import random
import sys
import os
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_PATH)
SIMULATION_DAYs = ["26", "25"]
for day in SIMULATION_DAYs:
# open het CSV-bestand voor lezen en schrijven
    with open(f'{ROOT_PATH}/datalog-gitignore/taxi-data/reduced-manhattan-taxi-201605{day}-peak.csv', 'r') as csv_in_file, open(f'{ROOT_PATH}/datalog-gitignore/taxi-data/priority-manhattan-taxi-201605{day}-peak.csv', 'w', newline='') as csv_out_file:

        # maak de csv reader en writer objecten
        csv_reader = csv.reader(csv_in_file)
        csv_writer = csv.writer(csv_out_file)

        # schrijf de eerste rij (header) van het CSV-bestand naar het uitvoerbestand
        header = next(csv_reader)
        csv_writer.writerow(header)

        # loop door elke rij in het CSV-bestand
        for row in csv_reader:

            # genereer een willekeurige integer tussen 0 en 3
            random_int = random.randint(0, 3)

            # voeg de willekeurige integer toe aan de rij
            row.append(str(random_int))

            # schrijf de gewijzigde rij naar het uitvoerbestand
            csv_writer.writerow(row)
