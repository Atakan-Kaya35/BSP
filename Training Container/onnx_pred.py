import numpy as np
import onnxruntime as ort
from pyified_resources import Standard_Vars

def test_onnx_model(model_path, sample_input):
    """
    Test an ONNX model with sample input
    Args:
        model_path: path to .onnx file
        sample_input: numpy array of shape (1, sequence_length, num_features)
    """
    # Create ONNX runtime session
    sess = ort.InferenceSession(model_path)
    
    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Run inference
    predictions = sess.run([output_name], {input_name: sample_input})
    
    return predictions[0]

# Example usage
if __name__ == "__main__":
    Standard_Vars.initialize(12)
    # 1. Load your trained ONNX model
    MODEL_PATH = "atakanka350@gmail.com_1.onnx"  # Replace with your model path
    
    # 2. Create test input (must match training shape)
    # Shape: (batch_size=1, sequence_length=12, features=3)
    # Features: [glucose, sin_time, cos_time]
    glucose_readings = [102, 104, 98, 94, 90, 84, 88, 93, 96, 95, 106, 120, 135, 150, 171]

    from datetime import datetime, timedelta

    # Assuming readings are every 5 minutes (adjust if different)
    sample_time = datetime.now()  # Starting time point
    time_points = [sample_time + timedelta(minutes=5*i) for i in range(len(glucose_readings))]

    # Calculate time features (sin/cos of minute-of-day)
    def time_to_features(t):
        minutes_of_day = t.hour * 60 + t.minute
        return [
            np.sin(2 * np.pi * minutes_of_day / 1440),  # 1440 min = 24hr
            np.cos(2 * np.pi * minutes_of_day / 1440)
        ]

    # Create the full input array (last 12 readings)
    sequence_length = 12
    input_data = []

    for i in range(len(glucose_readings) - sequence_length, len(glucose_readings)):
        # Scale glucose (assuming Standard_Vars.sc exists)
        scaled_glucose = Standard_Vars.sc.transform([[glucose_readings[i]]])[0][0]
        
        # Get time features
        time_feats = time_to_features(time_points[i])
        
        # Combine [glucose, sin_time, cos_time]
        input_data.append([scaled_glucose, time_feats[0], time_feats[1]])

    # Convert to numpy array with correct shape (1, 12, 3)
    model_input = np.array([input_data[-sequence_length:]], dtype=np.float32)

    
    # 3. Run prediction
    pred = test_onnx_model(MODEL_PATH, model_input)
    print(f"Predicted blood sugar: {(pred[0][0]*371+39):.1f} mg/dL")  # Single output value