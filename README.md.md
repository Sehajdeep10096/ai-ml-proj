# Study Mood Classifier

A lightweight machine learning tool that predicts a student's likely study mood — **Focused**, **Distracted**, or **Stressed** — based on five daily lifestyle inputs. It then offers a short, actionable suggestion tailored to that predicted mood.

---

## The Problem

Students often sit down to study without any awareness of how their daily habits are shaping their mental state. Sleep deprivation, excessive screen time, unmanaged stress, irregular study schedules, and a lack of breaks can all quietly erode concentration — but the connection between these habits and the resulting mood is rarely made explicit in the moment.

Without this awareness, students either push through ineffectively or disengage entirely, not knowing what corrective action to take.

---

## The Solution

This project builds a **Decision Tree Classifier** trained on a synthetically generated dataset that maps combinations of five behavioural features to one of three mood categories. At runtime, a student enters their daily stats and the model returns a predicted study mood along with a concrete suggestion.

The entire pipeline — dataset generation, model training, prediction, and feedback — runs in a single script with no prior setup required beyond installing the dependencies.

---

## How It Works

The pipeline has three stages:

### 1. Dataset Generation (`create_dataset`)

A CSV dataset of 60 rows is generated programmatically using `random`. Each row represents a hypothetical student's day, with the following features:

| Feature | Range | Description |
|---|---|---|
| `sleep_hours` | 2 – 9 | Hours of sleep the previous night |
| `screen_time` | 1 – 10 | Non-study screen time in hours |
| `stress` | 1 – 5 | Self-reported stress level |
| `study_time` | 0 – 6 | Hours spent studying |
| `breaks` | 0 – 6 | Number of breaks taken during study |

The mood label is assigned by a deterministic rule:

```
if sleep > 6 AND stress < 3 AND screen < 5 AND study >= 3  →  "Focused"
elif stress > 3 OR screen > 6 OR study == 2               →  "Stressed"
else                                                        →  "Distracted"
```

The dataset is written to `dataset.csv` and is only created once — subsequent runs reuse the existing file.

### 2. Model Training (`train_model`)

The CSV is loaded with `pandas`. The `mood` column (a string label) is integer-encoded using scikit-learn's `LabelEncoder`. The data is split 80/20 into training and test sets, and a `DecisionTreeClassifier` is fitted on the training portion.

Both the trained model and the label encoder are serialised to disk (`model.pkl`, `label_encoder.pkl`) using `pickle`, so they can be loaded independently at prediction time.

### 3. Prediction & Feedback (`pred_mood` + `Main`)

The user is prompted for their five daily values. These are assembled into a single-row `DataFrame` (preserving the column names the model was trained on) and passed to `model.predict()`. The numeric prediction is decoded back to a mood string via `LabelEncoder.inverse_transform`, and a short feedback message is printed.

---

## Project Structure

```
.
├── mood.py              # Main script — all logic lives here
├── dataset.csv          # Auto-generated on first run
├── model.pkl            # Serialised DecisionTreeClassifier
└── label_encoder.pkl    # Serialised LabelEncoder
```

---

## Dependencies

```
pandas
scikit-learn
numpy
```

Install with:

```bash
pip install pandas scikit-learn numpy
```

---

## Running the Script

```bash
python mood.py
```

**Example session:**

```
-:Study Mood Classifier:-
Dataset created (dataset.csv).
Enter details below (Only in numbers):
Hours of sleep (2-9): 7
Screen time(in hours) (1-10): 3
Stress level (1-5): 2
Study time(in hours) (0-6): 4
Breaks taken between study (0-6): 3

Predicted Study Mood: Focused
Good routine! Keep it up.
```

---

## Limitations & Possible Improvements

- **Synthetic data** — the dataset is rule-generated, so the model is essentially learning to approximate those same rules rather than real human behaviour. Replacing this with survey or self-report data would make predictions meaningfully more reliable.
- **Small dataset** — 60 rows is very small for even a Decision Tree. A larger dataset would reduce variance in the learned splits.
- **No test accuracy reported** — the training function splits data but never prints accuracy on the held-out test set. Adding a `model.score(x_test, y_test)` call would surface this immediately.
- **No input validation** — the script will crash if a user enters a value outside the expected range or a non-integer. Guard clauses or a loop-with-retry would harden the CLI.
- **Mood rule for `study == 2`** — labelling exactly two hours of study as "Stressed" (rather than "Distracted") is an unusual design choice that may introduce noise at the decision boundary.


