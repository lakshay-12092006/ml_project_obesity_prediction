import joblib
import pandas as pd
import gradio as gr

# Load the trained KNN model (created in your Google Colab notebook)
MODEL_PATH = "obesity_knn_model.pkl"

try:
    knn_model = joblib.load(MODEL_PATH)
except Exception as e:
    knn_model = None
    LOAD_ERROR = str(e)

# Encoding maps must match the maps used while training the model.
gender_map = {"Male": 0, "Female": 1}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
scc_map = {"no": 0, "yes": 1}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
favc_map = {"no": 0, "yes": 1}
smoke_map = {"no": 0, "yes": 1}
family_map = {"no": 0, "yes": 1}
mtrans_map = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4,
}

class_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

def predict_obesity(age, gender, height, weight, calc, scc, caec, favc,
                    fcvc, ncp, smoke, ch2o, family_history, faf, tue, mtrans):
    if knn_model is None:
        return "Model file not found or could not be loaded. Place obesity_knn_model.pkl beside app.py.\n" + LOAD_ERROR

    # Preserve the exact feature order used during model training.
    input_data = pd.DataFrame([[
        age,
        gender_map[gender],
        height,
        weight,
        calc_map[calc],
        scc_map[scc],
        caec_map[caec],
        favc_map[favc],
        fcvc,
        ncp,
        smoke_map[smoke],
        ch2o,
        family_map[family_history],
        faf,
        tue,
        mtrans_map[mtrans],
    ]], columns=[
        "Age", "Gender", "Height", "Weight", "CALC", "SCC", "CAEC", "FAVC",
        "FCVC", "NCP", "SMOKE", "CH2O", "family_history_with_overweight",
        "FAF", "TUE", "MTRANS"
    ])

    prediction = knn_model.predict(input_data)[0]
    predicted_class = class_map.get(int(prediction), str(prediction))
    return f"Predicted category: {predicted_class.replace('_', ' ')}"

with gr.Blocks(theme=gr.themes.Soft(), title="Obesity Level Prediction") as demo:
    gr.Markdown("# KNN-Based Obesity Level Prediction")
    gr.Markdown(
        "Enter the lifestyle and physical information below to obtain a model-estimated "
        "weight category. This is an educational ML demonstration, not a medical diagnosis."
    )

    with gr.Row():
        age = gr.Number(label="Age (years)", value=20, minimum=1, maximum=100)
        gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
        height = gr.Number(label="Height (m)", value=1.70, minimum=0.5, maximum=2.5)
        weight = gr.Number(label="Weight (kg)", value=70, minimum=10, maximum=300)

    with gr.Row():
        calc = gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="Alcohol consumption (CALC)", value="Sometimes")
        scc = gr.Dropdown(["no", "yes"], label="Monitoring calories (SCC)", value="no")
        caec = gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="Eating between meals (CAEC)", value="Sometimes")
        favc = gr.Dropdown(["no", "yes"], label="Frequent high-calorie food (FAVC)", value="yes")

    with gr.Row():
        fcvc = gr.Number(label="Vegetable intake frequency (FCVC)", value=2.0, minimum=0, maximum=3)
        ncp = gr.Number(label="Number of main meals (NCP)", value=3.0, minimum=0, maximum=10)
        smoke = gr.Dropdown(["no", "yes"], label="Smoking (SMOKE)", value="no")
        ch2o = gr.Number(label="Water intake (CH2O)", value=2.0, minimum=0, maximum=10)

    with gr.Row():
        family_history = gr.Dropdown(["no", "yes"], label="Family history of overweight", value="yes")
        faf = gr.Number(label="Physical activity frequency (FAF)", value=1.0, minimum=0, maximum=10)
        tue = gr.Number(label="Technology use time (TUE)", value=2.0, minimum=0, maximum=24)
        mtrans = gr.Dropdown(
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
            label="Transportation (MTRANS)",
            value="Public_Transportation"
        )

    predict_btn = gr.Button("Predict Weight Category", variant="primary")
    result = gr.Textbox(label="Model Output", interactive=False)

    predict_btn.click(
        fn=predict_obesity,
        inputs=[age, gender, height, weight, calc, scc, caec, favc, fcvc, ncp,
                smoke, ch2o, family_history, faf, tue, mtrans],
        outputs=result
    )

    gr.Markdown(
        "**Disclaimer:** The prediction is based on the model and dataset used for training. "
        "It should not be used to diagnose or guide medical treatment."
    )

if __name__ == "__main__":
    demo.launch()
