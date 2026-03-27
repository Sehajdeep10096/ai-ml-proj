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


# STEP 2: Training the model

def train_model():
    data = pnd.read_csv("dataset.csv")

    ler = LabelEncoder()
    data["mood"] = ler.fit_transform(data["mood"])

    X = data.drop("mood", axis=1)
    y = data["mood"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=8
    )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pkl.dump(model, open("model.pkl", "wb"))
    pkl.dump(ler, open("label_encoder.pkl", "wb"))

def pred_mood(sleep, screen, stress, study, brk):
    model = pkl.load(open("model.pkl","rb" ) )
    
    ler = pkl.load(open("label_encoder.pkl", "rb"))

    # Create a DataFrame with correct column names
    dfe = pnd.DataFrame([{
        "sleep_hours": sleep,
        "screen_time": screen,
        "stress": stress,
        "study_time": study,
        "breaks": brk
    }])

    pred_num = model.predict(dfe)[0]
    mood = ler.inverse_transform([pred_num])[0]
    return mood

# MAIN PROGRAM

def Main():
    print("-:Study Mood Classifier:-")

    create_dataset()
    train_model()

    print("Enter details below (Only in numbers):")

    sleep = int(input("Hours of sleep (2-9): "))
    screen = int(input("Screen time(in hours) (1-10): "))
    stress = int(input("Stress level (1-5): "))
    study = int(input("Study time(in hours) (0-6): "))
    brk = int(input("Breaks taken between study (0-6): "))

    mood = pred_mood(sleep, screen, stress, study, brk)

    print("Predicted Study Mood:", mood)

    if (mood == "Focused"):
        print("Good routine! Keep it up.")
    elif(mood == "Distracted"):
        print("Try reducing interruptions and stay focused.")
    else:
        print("You look stressed. Consider taking breaks and decrease screen time.")

# Calling the main function
Main()
