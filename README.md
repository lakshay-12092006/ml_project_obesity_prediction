# KNN-Based Obesity Level Prediction

A machine learning web application that uses a trained **K-Nearest Neighbors (KNN)** classifier to estimate a weight/obesity category from selected personal, eating-habit and lifestyle inputs. The interface is built with **Gradio** and the model is loaded from a saved `.pkl` file.

> **Disclaimer:** This is an educational demonstration, not a medical diagnostic tool. Predictions should not be used to make medical or treatment decisions.

## Project Overview

This project uses a KNN classification model trained on an obesity dataset. The app collects the model's input features and displays the predicted class.

### Predicted categories

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

## Features

- Interactive Gradio interface
- 16 input features, matching the model's training feature order
- Encodes categorical input values to match the notebook preprocessing
- Loads the trained model from `obesity_knn_model.pkl`
- Displays the predicted weight/obesity category

## Model and Inputs

The app uses the following input features:

| Feature | Description |
|---|---|
| `Age` | Age in years |
| `Gender` | Male or Female |
| `Height` | Height (metres) |
| `Weight` | Weight (kg) |
| `CALC` | Alcohol consumption frequency |
| `SCC` | Calorie-consumption monitoring |
| `CAEC` | Eating between meals |
| `FAVC` | Frequent consumption of high-calorie food |
| `FCVC` | Frequency of vegetable consumption |
| `NCP` | Number of main meals |
| `SMOKE` | Smoking status |
| `CH2O` | Water consumption |
| `family_history_with_overweight` | Family history of overweight |
| `FAF` | Physical activity frequency |
| `TUE` | Time spent using technology |
| `MTRANS` | Transportation method |

## Algorithm

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Number of neighbors (K):** 5
- **Distance metric:** Manhattan distance (L1), as configured during training (`p=1`)
- **Output:** One of seven encoded target categories

The app assumes that the saved model was trained with the feature encoding and feature order used in the Colab notebook.

## Project Structure

```text
Obesity-Prediction/
├── app.py
├── obesity_knn_model.pkl
└── README.md
```

## Requirements

- Python 3.8+ (recommended)
- pandas
- scikit-learn
- joblib
- Gradio

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Replace the repository URL and folder name with your own.

### 2. Install dependencies

```bash
pip install gradio pandas scikit-learn joblib
```

### 3. Add the trained model

Place `obesity_knn_model.pkl` in the same folder as `app.py`.

The model file is created in the Colab notebook using:

```python
import joblib
joblib.dump(knn_model, "obesity_knn_model.pkl")
```

Download the file from Colab and add it to your project folder.

### 4. Start the app

```bash
python app.py
```

Open the local Gradio URL shown in the terminal (typically `http://127.0.0.1:7860`).

## Dataset

The model was trained in Google Colab using the obesity dataset referenced in the notebook:

- Dataset source: [Obesity dataset on GitHub](https://github.com/chandanXP/DataScienceDocs)

Please review the dataset's license and attribution requirements before redistributing it or the trained model.

## Future Improvements

- Add feature scaling and evaluate its impact on KNN performance
- Evaluate with cross-validation, confusion matrix and per-class metrics
- Add validation for inputs and clearer prediction explanations
- Improve the interface and deploy it online
- Test fairness, privacy and performance before any real-world use

## Author

**[Your Name]**  
B.Tech CSE | Machine Learning Project

---

If you found this project useful, consider starring the repository.
