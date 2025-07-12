import traceback
import numpy as np
import logging
#from pyified_resources import Models
from pyified_resources import Standard_Vars

class Pred_Tools():

    @staticmethod
    def many_model_predict(values, models=None):
        """
        Predicts the values using a bag of ONNX models
        Appends the current time kept by the system

        Args:
            values: numpy array of shape (1, seq_len, input_dim) as a py list
            models: list of onnxruntime.InferenceSession instances

        Returns:
            mean_pred: list containing the mean predicted blood sugar value, shape (input_dim)
            preds: list of individual predictions from each model, 2D array ready for minmaxsc
        """
        print(Standard_Vars.current_time)
        Standard_Vars.current_time = (Standard_Vars.current_time + 5) % 1440

        preds = []
        for model in models:
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name

            # ONNX requires float32
            result = model.run([output_name], {input_name: values.astype(np.float32)})
            
            # Shape: result[0] should be (1, 1) or (1,) depending on model
            preds.append([result[0][0][0]])  # Extract scalar from shape (1, 1)

        individual_predictions = [p[0] for p in preds]
        only_blood_sugar = Standard_Vars.sc_sugar.inverse_transform([[np.mean(individual_predictions)]])
        predicted_blood_sugar = Standard_Vars.sc.transform([[only_blood_sugar[0][0], np.sin(2 * np.pi * Standard_Vars.current_time / 1440), np.cos(2 * np.pi * Standard_Vars.current_time / 1440)]])[0]
        individual_predictions = [[i] for i in individual_predictions]
        return predicted_blood_sugar, individual_predictions


    def pred_next_arbitrary(past_values, wanted_history = Standard_Vars.FIVE_MIN_INTERVAL, interval_num = 3, models = None):
        """
        Predicts the next arbitrary number of bs values at given context

        Args: 
            past_values: any length of bs values in 2D float [blood sugar, TOD], py list
            wanted_history: the number of most recent values to be used for the prediction, important for the input style of the pred model
            interval_num: the number of bs values to be predicted, int
            models: the bag of model to do the predicting, read .onnx files turned into class objects
        
        Returns:
            Predictions: 2D array of interval_num many predicted values, shape: (1, seq_len, input_dim)
            indic_data: the individual predictions of the models to be further used in indication analysis
        """
        predictions = []
        indic_data = [[]] * len(models)

        # modify the simple array into scaled pd dataframe
        past_values = np.array(past_values)  # Should already be 2D from app.py
        past_values = Standard_Vars.sc.transform(past_values)

        # get only what is required for the model
        X_test = past_values[len(past_values) - wanted_history :]

        # configure X_test to fit the model
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (1, Standard_Vars.REG_SHAPE, Standard_Vars.INPUT_DIM))

        # make prediction
        mean_pred, individual_preds = Pred_Tools.many_model_predict(X_test, models = models)
        predictions.append(mean_pred)
        
        # prepare indicator values
        indic_data = Standard_Vars.sc_sugar.inverse_transform(individual_preds)

        # append the list
        np_predictions = np.array([mean_pred])             # shape (2,)
        np_predictions = np_predictions.reshape(1, 1, -1)      # shape (1, 1, dim)
        X_test = np.append(X_test, np_predictions, axis=1) # shape becomes (1, 13, dim)

        # repeats desired interval number - 1 times from here
        # get list ready for fiting model
        for i in range(interval_num - 1):
            X_test = X_test[:, 1:, :]

            # make prediction
            mean_pred, individual_preds = Pred_Tools.many_model_predict(X_test, models = models)
            predictions.append(mean_pred)
            
            # prepare indicator values
            indic_data = Standard_Vars.sc_sugar.inverse_transform(individual_preds)

            # append the list
            np_predictions = np.array([mean_pred])             # shape (2,)
            np_predictions = np_predictions.reshape(1, 1, -1)      # shape (1, 1, 2)
            X_test = np.append(X_test, np_predictions, axis=1) # shape becomes (1, 13, 2)

        predictions = Standard_Vars.sc_sugar.inverse_transform(predictions)

        return predictions, indic_data

class Model_Assessment():

    def indicator_recognizer(values, score_list = None):
        """
        Assesses the indicator values, ready to be sent

        Args: 
            values: the set of all prediction values from all models in order in 4D array form [[[[]],[[]]],[[[]]...]]
            score_list: a 2D list [[a,b,c], ...] with scores of the models the predicted value belongs to
        
        Returns:
            Three digit indicator value with (extreme value indic, plateau indic, trend change indic) abc based on the scores of models good in that field.
                an int vale between low 200 to 0 with meaning to each decimal step
        """
        try:
            global scores

            if score_list == None:
                score_list = scores

            # Calculate the mean of the input values
            mean = sum(sum(x) for x in values) / len(values)

            # Initialize an integer to store the indicators
            indicator_list = 0

            # Set thresholds and bounds
            lower_bound = 80
            upper_bound = 200
            proficient_model_score_threshold = 0.59
            indicative_value_threshold = 13

            # Iterate through the scores and values
            for i, score in enumerate(score_list):
                # Check extreme values indicator by first checking model proficiency on the topic 
                # Then checks if the proficient model made an expert extreme value pred
                # Adds 100 if low expected, 200 if high expected
                if score[0] > proficient_model_score_threshold:
                    if values[i][-1][0][0][0] < lower_bound:
                        indicator_list += 100
                    elif values[i][-1][0][0][0] > upper_bound:
                        indicator_list += 200

                # Check plateau indicator: model proficiency -> checks if the pred is stable
                # adds 10 if it is
                if score[1] > proficient_model_score_threshold:
                    if abs(mean - values[i][-1][0][0]) <= indicative_value_threshold:
                        indicator_list += 10

                # Check trend change indicator:  model proficiency -> checks if the pred is deviating from trend
                # If there is a possible trend downward adds 1, if upward adds 2
                if score[2] > proficient_model_score_threshold:
                    if abs(mean - values[i][-1][0][0]) >= indicative_value_threshold:
                        if (mean - values[i][-1][0][0]) > 0:
                            indicator_list += 1
                        elif (mean - values[i][-1][0][0]) < 0:
                            indicator_list += 2

            return indicator_list
        except Exception as e:
            error_trace = traceback.format_exc()
            logging.error(error_trace)
            return 0



class Communication():
    def jsonBuilder(values, last_dexcom_instance, indicators):
        """
        Creates the json dictionary format
        Also updates the SQL database as it is the best time to do so

        Args: 
            values: actual bs values with the mean bs values at the end, 2D list [[dim], [dim], ...] 
            last_dexcom_instance: the last Dexcom value with dexcom trend info, Dexcom class object
            indicators: the indicator values to be sent , low 200s to 0 int value
        
        Returns:
            .json Format: [safeness (bool), trend (in 2 digits), befores(in 9 digits), afters (in 9 digits), befores1(in 9 digits), afters (in 9 digits), befores2(in 9 digits), afters (in 9 digits), indicators (in 3 digits / trend change, stable, plateau)]
            db Format: safeness (bool), trend (in 2 digits), befores(in 9 digits), afters (in 9 digits), indicators (in 3 digits / trend change, stable, plateau)
        """
        answer = []
        
        try:
            # the safe / unsafe cell
            if values[-1][0] < 80 or values[-1][0] > 200:
                answer.append(False)
            else:
                answer.append(True)
            
            des = last_dexcom_instance.trend_description
            
            if des == "rising quickly":
                answer.append(48)
            elif  des == "rising":
                answer.append(40)
            elif des == "rising slightly":
                answer.append(32)
            elif des == "steady":
                answer.append(24)
            elif des == "falling slightly":
                answer.append(16)
            elif des == "falling":
                answer.append(8)
            elif des == "falling quickly":
                answer.append(0)
            else:
                answer.append(56)
            
            difference = values[-1][0] - values[-4][0]
            
            if difference > 30:
                answer[1] += 6
            elif difference > 20:
                answer[1] += 5
            elif difference > 10:
                answer[1] += 4
            elif difference > -10:
                answer[1] += 3
            elif difference > -20:
                answer[1] += 2
            elif difference > -30:
                answer[1] += 1
            elif difference <= -30:
                answer[1] += 0
            else:
                answer[1] += 7
            
            return {"safeness" : answer[0], "trend" : answer[1], "befores" : (int)(values[-6][0] * 10**6 + values[-5][0] * 10**3 + values[-4][0]), "befores1" : (int)(values[-9][0] * 10**6 + values[-8][0] * 10**3 + values[-7][0]), "befores2" : (int)(values[-12][0] * 10**6 + values[-11][0] * 10**3 + values[-10][0]), "afters" : int(values[-3][0]) * 10**6 + int(values[-2][0]) * 10**3 + int(values[-1][0]), "indicators" : indicators}
        
        except Exception as e: 
            logging.exception(f"Error in jsonDictBuilder: {e}")
            return {}
