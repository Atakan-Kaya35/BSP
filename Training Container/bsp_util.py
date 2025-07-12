import json
import os
import traceback
import zipfile
import shutil
import numpy as np
import pandas as pd
import logging
from pyified_resources import Standard_Vars
from bsp_cloud_lib import Cloud_Storage
from keras.callbacks import EarlyStopping
from config import Config
from pathlib import Path
import tensorflow as tf
import tf2onnx

class Model_Assessment():
    @staticmethod
    def accuracy_finder(predicted, real, percentage_error_threshold=0.03, big_error_treshold=5):
        """
        Calculates accuracy between two sets

        Args:
            prediction: the predicted values, as a list of dimension Standard_Vars.DIM
            real: the actual values in the same time interval
            percentage_error_threshold: percentage deviation for a pred to be considered true

        Returns:
            the accuracy coefficient between two sets
        """
        # Calculate the absolute percentage error for each prediction
        # absolute_error = np.abs((predicted - real))
        absolute_percentage_error = np.abs((predicted - real) / real)

        # Count the number of accurate predictions based on the threshold
        accurate_predictions = np.sum(absolute_percentage_error <= percentage_error_threshold)
        # big_accurate_predictions = np.sum(absolute_error <= big_error_treshold)

        # Calculate the percentage of accurate predictions
        accuracy_coef = (accurate_predictions / len(real))

        return accuracy_coef
    
    @staticmethod
    def model_accuracy_finder(model, X_test, y_test):
        """Get the real blood sugar values"""
        """ predicted_blood_sugar = model.predict(X_test)
        return Model_Assessment.accuracy_finder(predicted_blood_sugar, y_test) """
        predicted_blood_sugar = model.predict(X_test)
        # reshape for scaler: (N,1)
        predicted_blood_sugar = predicted_blood_sugar.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # reverse transform both
        predicted_blood_sugar = Standard_Vars.sc.inverse_transform(predicted_blood_sugar)
        y_test = Standard_Vars.sc.inverse_transform(y_test)

        return Model_Assessment.accuracy_finder(predicted_blood_sugar, y_test)


        
    @staticmethod
    def model_score_generator(models):
        """
        Gererates the scores for all models in a bag of models [extreme, plateau, trend change]

        Args: 
            models: a bag of models to be evaluated
        
        Returns:
            2D List of scores in the form: 
            [extreme values score, plateau score, trend change score] 
            for every model in seqiential order of the model mashup
        """
        scores = []
        evaluation_datasets = Standard_Vars.evaluation_datasets

        for model in models:
            score = []
            for evaluation_dataset in evaluation_datasets:
                indication_rating = 0
                for i in range(len(evaluation_dataset)):
                    # Get the full sequence including future values we want to predict
                    full_sequence = evaluation_dataset[i]
                    
                    # Take first Standard_Vars.FIVE_MIN_INTERVAL points as input
                    input_data = full_sequence[:Standard_Vars.FIVE_MIN_INTERVAL]
                    input_data = np.array(input_data)  # Convert to numpy array
                    
                    # Scale the input data
                    X_test = input_data.reshape(1, Standard_Vars.REG_SHAPE, Standard_Vars.INPUT_DIM)
                    
                    # Make three predictions recursively
                    predictions = []
                    current_input = X_test.copy()
                    
                    for _ in range(3):
                        pred = model.predict(current_input)[0][0]
                        predictions.append(pred)
                        
                        # Create new input for next prediction
                        # Original: for no tod testing
                        #new_point = np.array([[pred, 0.5]])  # Using time=0 for future points
                        new_point = np.array([Standard_Vars.medianalyze_point(pred)])
                        current_input = np.append(
                            current_input[:, :-1, :],
                            [new_point],
                            axis=1
                        )
                    
                    # Get the actual third value from the dataset
                    actual_third_value = full_sequence[Standard_Vars.FIVE_MIN_INTERVAL + 2]
                    
                    # Create dummy array for inverse transform of prediction
                    # Original: for no tod testing
                    #dummy_pred = np.array([[predictions[2], 0.5]])
                    dummy_pred = np.array([Standard_Vars.medianalyze_point(predictions[2])])
                    pred_glucose = Standard_Vars.sc.inverse_transform(dummy_pred)[0][0]
                    
                    # Calculate accuracy for the third prediction only
                    accuracy_score = Model_Assessment.accuracy_finder(
                        np.array([pred_glucose]), 
                        np.array([actual_third_value[0]])
                    )
                    indication_rating += accuracy_score
                    
                score.append(indication_rating / len(evaluation_dataset))
            scores.append(score)
        return scores


class Model_Creation():
    @staticmethod
    def train_send_model(
        username,
        source_csv_file_name=None,
        requested_num_of_models=2,
        epochs=2,
        batch_size=24,
        remaining_tries=7,
        acceptable_acc_score=0.10,
        num_of_layers=3
        ):
        """
        Creates and ships a zip file to the cloud with
        a custome model with adjustably acceptable statistics, its context scores 
        and the data it was trained on; proceeds to delete the zip file and the files in it

        Args:
            username: the username of the user
            source_csv_file_name: the name of the data file in case it is not same with the username
            requested_num_of_models: how many models to train and store
            epochs: how many epochs to train
            batch_size: the size of batch to train with
            remaining_tries: how many attempts are allowed before giving up
            acceptable_acc_score: the lowest acceptable validation score for a model

        Returns:
            To The Cloud:
                Zip file containing following: custome model file, its context scores, data it was trained on
            As code:
                Dictionary type API return-ready json message
                API status code
        """
        try:
            EPOCHS = epochs
            BATCH_SIZE = batch_size
            num_of_model_till_done = requested_num_of_models
            num_models_accepted = 0
            CSV_METADATA_SKIP = 25
            TIME_COL = "Zaman damgası (GG-AA-YYYY/ss:dd:sn)"
            GLUCOSE_COL = "Glikoz Değeri (mg/dL)"
            TEST_SPLIT_RATIO = 0.1  # e.g., 15% of data for final testing
            VAL_SPLIT_RATIO = 0.1            
            EARLY_STOP_PATIENCE = 3
            
            # default file name is the username of user
            if source_csv_file_name is None:
                source_csv_file_name = f'{username}.csv'

            # Data Loading and Preprocessing
            def load_and_preprocess_data():
                df_raw = pd.read_csv(source_csv_file_name, sep=";")
                
                # Replace categorical glucose values
                # 41 to prevent division by 0
                df_raw.iloc[CSV_METADATA_SKIP:, 7] = df_raw.iloc[CSV_METADATA_SKIP:, 7].replace({
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
                df_clean = df_raw.dropna(subset=[GLUCOSE_COL, "sin_time", "cos_time"]).iloc[CSV_METADATA_SKIP:]                

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

           # 1. Split into Training+Validation set and Test set (preserving temporal order)
            test_size = int(len(X_sequences_all) * TEST_SPLIT_RATIO)
            X_train_val = X_sequences_all[:-test_size]
            y_train_val = y_sequences_all[:-test_size]
            X_test = X_sequences_all[-test_size:]
            y_test = y_sequences_all[-test_size:]

            # 2. Split Training+Validation set into Training set and Validation set (preserving temporal order)
            val_size = int(len(X_train_val) * VAL_SPLIT_RATIO)
            X_train = X_train_val[:-val_size]
            y_train = y_train_val[:-val_size]
            X_val = X_train_val[-val_size:]
            y_val = y_train_val[-val_size:]

            # Importing the Keras libraries and packages
            #from tensorflow.keras import Input, Model
            from keras.layers import Dense
            from keras.layers import GRU
            from keras.layers import Dropout
            from keras.models import Sequential
            from keras.losses import MeanSquaredError

            while num_of_model_till_done > 0 and remaining_tries > 0:
                
                # --- Model Architecture ---
                regressor = Sequential(name="blood_sugar_predictor")
                
                # Input Layer (explicitly named)
                regressor.add(tf.keras.layers.InputLayer(
                    input_shape=(Standard_Vars.FIVE_MIN_INTERVAL, Standard_Vars.INPUT_DIM),
                    name="input_layer"
                ))

                # GRU Layers (EXACTLY AS IN YOUR ORIGINAL CODE)
                # First GRU
                regressor.add(GRU(
                    units=50,
                    return_sequences=True,
                    name="GRU_1"
                ))
                regressor.add(Dropout(0.2, name="dropout_1"))

                # Additional GRUs
                for i in range(max(0, num_of_layers - 2)):
                    regressor.add(GRU(
                        units=50,
                        return_sequences=True,
                        name=f"GRU_{i+2}"
                    ))
                    regressor.add(Dropout(0.2, name=f"dropout_{i+2}"))

                # Final GRU (no return_sequences)
                regressor.add(GRU(
                    units=50,
                    return_sequences=False,  # VITAL/Critical for single-step prediction
                    name="GRU_final"
                ))
                regressor.add(Dropout(0.2, name="dropout_final"))

                # Output Layer
                regressor.add(Dense(units=1, name="output"))

                # --- Training (Unchanged from your original) ---
                regressor.compile(optimizer='adam', loss=MeanSquaredError())
                early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
                regressor.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop])

                # check if the model is acceptable
                curr_model_acc = Model_Assessment.model_accuracy_finder(regressor, X_test, y_test)
                if curr_model_acc > acceptable_acc_score:
                    # Ensure tmp directory exists
                    Config.TMP_DIR.mkdir(parents=True, exist_ok=True)
                    
                    print("Model ACCEPTED with accuracy:", curr_model_acc)
                    num_of_model_till_done -= 1
                    num_models_accepted += 1
                        
                    # if the model in hand passed the acceptable threshold
                    # TODable: HighJacked
                    scores = Model_Assessment.model_score_generator([regressor])
                    scores = [score for i in scores for score in i]

                    # Makes the regressor and the scores into h5 and txt files respectively and send them to the cloud
                    scores = {
                        "extreme values": scores[0], 
                        "plateau": scores[1], 
                        "trend change": scores[2]
                    }
                    score_path = Config.TMP_DIR / f"{username}_{num_models_accepted}_scores.json"
                    with open(score_path, 'w') as f:
                        json.dump(scores, f)  
                        
                    #input_shape = (1, Standard_Vars.FIVE_MIN_INTERVAL, Standard_Vars.INPUT_DIM)  # Adjust the last number based on your actual input features
                    #input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')]

                    onnx_model_path = Config.TMP_DIR / f"{username}_{num_models_accepted}.onnx"
                    # 🔒 Freeze all layers to inference mode
                    regressor.trainable = False
                    for layer in regressor.layers:
                        layer.trainable = False

                    # Convert to Functional API for proper ONNX export
                    input_tensor = tf.keras.Input(
                        shape=(Standard_Vars.FIVE_MIN_INTERVAL, Standard_Vars.INPUT_DIM),
                        name="input"
                    )
                    output_tensor = regressor(input_tensor)
                    functional_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

                    # Export to ONNX
                    tf2onnx.convert.from_keras(
                        functional_model,
                        input_signature=[tf.TensorSpec(
                            shape=(1, Standard_Vars.FIVE_MIN_INTERVAL, Standard_Vars.INPUT_DIM),
                            dtype=tf.float32,
                            name="input"
                        )],
                        opset=13,
                        output_path=str(onnx_model_path)
                    )
                    
                    print(regressor.summary())  # Should show all GRU layers

                else:
                    print("Model REJECTED with accuracy:", curr_model_acc)
                                    
                remaining_tries -= 1

            if num_models_accepted > 0:
                # Create a zip file containing both the regressor and the scores
                zip_filename = f"{username}_data.zip"
                output_zip_path = Config.TMP_DIR / zip_filename

                metadata = {"number of models": num_models_accepted}
                metadata_filename = f"{username}_metadata.json"
                metadata_file_path = Config.TMP_DIR / metadata_filename
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f)
                    
                # Zip everything from absolute paths
                with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    for i in range(1, num_models_accepted + 1):
                        # Paths
                        #model_dir = Config.TMP_DIR / f"{username}_{i}_model"
                        #model_zip_path = Config.TMP_DIR / f"{username}_{i}_model.zip"
                        score_path = Config.TMP_DIR / f"{username}_{i}_scores.json"
                        onnx_path = Config.TMP_DIR / f"{username}_{i}.onnx"

                        # Zip SavedModel folder into .zip
                        #shutil.make_archive(str(model_zip_path).replace('.zip', ''), 'zip', model_dir)

                        # Add zipped model + score file to output zip
                        #zipf.write(model_zip_path, arcname=model_zip_path.name)
                        zipf.write(score_path, arcname=score_path.name)
                        zipf.write(onnx_path, arcname=onnx_path.name)

                    zipf.write(metadata_file_path, arcname=metadata_file_path.name)
                    source_csv_path = Path(Config.TMP_DIR) / source_csv_file_name
                    zipf.write(source_csv_path, arcname=Path(source_csv_file_name).name)
                    
                # Upload the zip file to the cloud
                if not Config.IS_LOCAL:                
                    Cloud_Storage.upload_to_s3(f"{username}/{zip_filename}", str(output_zip_path))

                # Delete the local files, but not needed as the instance, for the most recent single person 
                # synchronus implementation, will shut the insnace down anyway

                return {"Success": "File created and uploaded successfully!"}, 200

            return {"Failure": "No acceptable models were attained in training!"}, 400
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error occurred: {error_details}", error_details)
            return {"error": str(e)}, 500
            
    @staticmethod
    def full_model_creation(
        username, 
        num_of_models=2, 
        epochs=2, 
        batch_size=24, 
        remaining_tries=7, 
        acceptable_acc_score=0.10, 
        num_of_layers=3
    ):
        """
        Parameters
        ----------
        username : string
            username to obtaşn path necessary to access the S3 files.
        num_of_models : TYPE, optional
            DESCRIPTION. The default is 2.
        epochs : TYPE, optional
            DESCRIPTION. The default is 2.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 24.
        remaining_tries : TYPE, optional
            DESCRIPTION. The default is 7.
        acceptable_acc_score : TYPE, optional
            DESCRIPTION. The default is 0.10.
        num_of_layers : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        dict
            DESCRIPTION.
        int
            DESCRIPTION.

        """
        try:
            if Config.IS_LOCAL:
                input_csv_path = f".\{username}.csv"
            else:
                os.makedirs(Config.TMP_DIR, exist_ok=True)
                input_csv_path = Config.TMP_DIR / f"{username}.csv"
                Cloud_Storage.download_from_s3(f"{username}/{username}.csv", str(input_csv_path))

            # Call with extended params
            Model_Creation.train_send_model(
                username,
                input_csv_path,
                requested_num_of_models=num_of_models,
                epochs=epochs,
                batch_size=batch_size,
                remaining_tries=remaining_tries,
                acceptable_acc_score=acceptable_acc_score,
                num_of_layers=num_of_layers
            )

            return {"Hurray!": "All seems fine"}, 200

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error occurred: {error_details}", error_details)
            return {"error": str(e)}, 500