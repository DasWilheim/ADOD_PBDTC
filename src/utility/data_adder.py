import os
import sys
import csv
import math
import random
 


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_PATH)

SIMULATION_DAYs = ["26"]
for day in SIMULATION_DAYs:
    with open(f'{ROOT_PATH}/datalog-gitignore/taxi-data/20160525-flat1.1.csv', 'r') as csv_in_file, open(f'{ROOT_PATH}/datalog-gitignore/taxi-data/aaaa.csv', 'w', newline='') as csv_out_file:

        csv_reader = csv.reader(csv_in_file)
        csv_writer = csv.writer(csv_out_file)

        header = next(csv_reader)
        csv_writer.writerow(header)

        period = 2000  # Adjust this value to change the wave period

        for i, row in enumerate(csv_reader):
            # Calculate the sine wave value for the current index
            sine_wave_value = math.sin(2 * math.pi * i / period)

            # Assign a priority based on the sine wave value
            probability_priority_1 = (0.14 * sine_wave_value) + 0.3

            # Assign a priority based on the probability
            if random.random() < probability_priority_1:
                priority = 1
            else:
                priority = 2

            row[-1] = str(priority)  # Replace the priority value in the row

            csv_writer.writerow(row)
