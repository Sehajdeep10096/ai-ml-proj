import pandas as pnd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pickle as pkl
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# STEP 1: Create dataset if it doesn't already exist

def create_dataset():
    if os.path.exists("dataset.csv"):
        return  # Don't recreate every time

    rows = []
    for _ in range(60):
        sleep = random.randint(2, 9)
        screen = random.randint(1, 10)
        stress = random.randint(1, 5)
        study = random.randint(0, 6)
        brk = random.randint(0, 6)

        # Logic to give each row a mood
        if(sleep > 6 and stress < 3 and screen < 5 and study>= 3):
            mood = "Focused"
        elif stress > 3 or screen > 6 or study == 2:
            mood = "Stressed"
        else:
            mood = "Distracted"

        rows.append([sleep, screen, stress, study, brk, mood])

    df = pnd.DataFrame(rows, columns=[
        "sleep_hours", "screen_time", "stress", "study_time", "breaks", "mood"
    ])

    df.to_csv("dataset.csv", index=False)
    print("Dataset created (dataset.csv).")
