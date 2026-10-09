# Sonar Rock vs Mine Classification

A beginner machine learning project created in Google Colab. It uses logistic regression to classify sonar readings as either a rock (`R`) or a metal cylinder (`M`, used as the mine class in this dataset).

## Project overview

Each example contains 60 numeric sonar features. The notebook walks through:

1. Loading and exploring the dataset
2. Separating feature values (`X`) from labels (`Y`)
3. Splitting the data into training and test sets
4. Training a logistic regression classifier
5. Measuring accuracy on the training and test sets
6. Making a prediction for one sonar reading

## Dataset

This project uses the [UCI Connectionist Bench Sonar dataset](https://archive.ics.uci.edu/dataset/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks). It contains 208 examples and 60 features per example. The labels are `R` for rock and `M` for a metal cylinder.

## Model and evaluation

The notebook uses `LogisticRegression` from scikit-learn. It holds out 10% of the examples for testing with `train_test_split(test_size=0.1, stratify=Y, random_state=1)`.

Results from the current notebook run:

- Training accuracy: **83.4%**
- Test accuracy: **76.2%**

These scores come from one split of a small dataset. They can change with a different split or model setup, and should not be treated as a guarantee of performance on real sonar data.

## Run the notebook

1. Open `sonar_rock_mine_classification.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
2. Make the dataset available using the same method and file path used in the notebook. If running in Colab, upload the dataset or mount Google Drive if the notebook expects it there.
3. Run the cells from top to bottom.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn

If you run the notebook locally, install the packages with:

```bash
pip install numpy pandas scikit-learn
```

## Repository files

- `sonar_rock_mine_classification.ipynb` — Colab notebook containing the code, explanations, and results.
- `README.md` — project description and run instructions.

## What I learned

- How feature data and labels are used in supervised learning
- Why data is split into training and test sets
- How to train and use a logistic regression classifier
- How to compare predicted labels with known labels to calculate accuracy
- Why evaluation on held-out examples matters

## Possible next steps

- Add a confusion matrix and classification report (precision, recall, and F1-score)
- Use cross-validation for a more stable performance estimate
- Compare logistic regression with another classifier
