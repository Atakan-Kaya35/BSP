import numpy as np
import pandas as pd
import onnxruntime as ort
from keras.models import load_model
from pyified_resources import Standard_Vars
from bsp_util import Model_Assessment

def evaluate_model_accuracy(model_type, model_path, X_test, y_test):
    preds = []

    if model_type == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        for sample in X_test:
            sample_input = np.expand_dims(sample, axis=0).astype(np.float32)  # shape: (1, 12, 3)
            pred = sess.run([output_name], {input_name: sample_input})[0][0]
            preds.append(pred)
    
    elif model_type == "keras":
        from keras.models import load_model
        model = load_model(model_path)
        preds = model.predict(X_test)

    preds = np.array(preds).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Reverse scale
    preds = Standard_Vars.sc.inverse_transform(preds)
    y_test = Standard_Vars.sc.inverse_transform(y_test)

    return Model_Assessment.accuracy_finder(preds, y_test)

# --- Example usage ---
if __name__ == "__main__":
    Standard_Vars.initialize(12)

    # Load X_test and y_test exactly as you do in training pipeline
    GLUCOSE_COL = "Glikoz Değeri (mg/dL)"
    TIME_COL = "Zaman damgası (GG-AA-YYYY/ss:dd:sn)"

    
    # Data Loading and Preprocessing
    def load_and_preprocess_data():
        df_raw = pd.read_csv("atakanka350@gmail.com.csv", sep=";")
        
        # Replace categorical glucose values
        # 41 to prevent division by 0
        df_raw.iloc[25:, 7] = df_raw.iloc[25:, 7].replace({
            "Yüksek": 400, 
            "Düşük": 40
        })
        df_raw[GLUCOSE_COL] = pd.to_numeric(df_raw[GLUCOSE_COL], errors="coerce")
        
        # Convert datetime and extract time-of-day
        df_raw[TIME_COL] = pd.to_datetime(df_raw[TIME_COL], errors='coerce')
        df_raw["hour"] = df_raw[TIME_COL].dt.hour
        df_raw["minute"] = df_raw[TIME_COL].dt.minute
        df_raw["sin_time"] = np.sin(2 * np.pi * (df_raw["hour"] * 60 + df_raw["minute"]) / 1440)
        df_raw["cos_time"] = np.cos(2 * np.pi * (df_raw["hour"] * 60 + df_raw["minute"]) / 1440)
        
        #TODO: this is not acceptable since missing in time series cannot just be dropped
        # Drop rows with missing values
        df_clean = df_raw.dropna(subset=[GLUCOSE_COL, "sin_time", "cos_time"]).iloc[25:]                

        # TODO: Original is:
        #return df_clean[[GLUCOSE_COL, "time_of_day"]].values
        # below is the modification to exclude the time of the day value for testing
        return df_clean
        #[[GLUCOSE_COL]].values  # remove TOD

    # Create sequences with time features
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            # if you but 1 instead of 0 you predict the TOD
            y.append(data[i, 0])  # Glucose value is at index 0
        return np.array(X), np.array(y)

    # Main execution
    df_clean = load_and_preprocess_data()
    
    glucose_scaled = Standard_Vars.sc.transform(df_clean[[GLUCOSE_COL]])
    sin_scaled = Standard_Vars.sc_time.transform(df_clean[["sin_time"]])
    cos_scaled = Standard_Vars.sc_time.transform(df_clean[["cos_time"]])
    
    # Combine features: [glucose, sin_time, cos_time]
    data = np.hstack((glucose_scaled, sin_scaled, cos_scaled))

    # Create sequences
    X_sequences_all, y_sequences_all = create_sequences(data, Standard_Vars.FIVE_MIN_INTERVAL)


    keras_acc = evaluate_model_accuracy("keras", "atakanka350@gmail.com_1.h5", X_sequences_all, y_sequences_all)
    onnx_acc = evaluate_model_accuracy("onnx", "atakanka350@gmail.com_2.onnx", X_sequences_all, y_sequences_all)

    print(f"Keras H5 Accuracy:  {keras_acc:.4f}")
    print(f"ONNX Accuracy:      {onnx_acc:.4f}")
