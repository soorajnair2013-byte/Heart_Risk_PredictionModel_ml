# Import Data Manipulation Libraries
import numpy as np
import pandas as pd 


def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/deveshdubey18/HeartAttack_PredictionModel/refs/heads/main/data/Heart%20Attack.csv')
    return df

