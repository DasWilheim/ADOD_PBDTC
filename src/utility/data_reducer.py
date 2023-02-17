#Data reducer
import pandas as pd
import numpy as np
import os
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SIMULATION_DAYs = ["26", "25"]
for day in SIMULATION_DAYs:
    df = pd.read_csv(f"{ROOT_PATH}/datalog-gitignore/taxi-data/manhattan-taxi-201605{day}-peak.csv")
    print(len(df))
    #only keep the 153th row
    df = df.iloc[::100]
    print(len(df))
    #save to csv
    df.to_csv(f"{ROOT_PATH}/datalog-gitignore/taxi-data/reduced-manhattan-taxi-201605{day}-peak.csv", index=False)

