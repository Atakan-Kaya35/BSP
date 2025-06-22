import json
import os
import traceback
import zipfile
from pydexcom import Dexcom
from keras.models import load_model
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from pyified_resources import Models
from pyified_resources import Standard_Vars
from bsp_cloud_lib import Cloud_Storage



class Pred_Tools():

    @staticmethod
    def many_model_predict(values, models = Models.MODEL_MIX):
        """
        Predicts the values using a bag of models

        Args:
            values: set of values to make pred with
            models: the set of models desired to be used
        
        Returns:
            Single value of the prediction:
            The individual predictions in a 2D array [[],[]]
        """
        preds = []
        for model in models:
            # due to the [blood sugar, TOD], TOD is set to the next 5 minutes
            # TODO: Should it be 2 * values[0][1] - values[1][1] instead, is the time going from right to left or left to right/ TIME INCREASES AND THE PREDİCTİON IS MADE SO THE FİRST ELEMNT
            # IN THE VALUES LİST IS THE FARTHEST FROM THE PRESENT
            a = model.predict(values)
            preds.append([model.predict(values)[0][0] 
                          # Original: tod exemption
                          #,2 * values[0][-1][1] - values[0][-2][1]
                          ])
        predicted_blood_sugar = [np.mean(preds[:][0], axis=0)
                                 # Original: tod exemption
                                 #, preds[0][1]
                                 ]
        return predicted_blood_sugar, preds


    def pred_next_arbitrary(past_values, wanted_history = Standard_Vars.FIVE_MIN_INTERVAL, interval_num = 3, models = Models.MODEL_MIX):
        """
        Predicts the next arbitrary number of bs values at given context

        Args: 
            past_values: any length of bs values in 2D float [blood sugar, TOD]
            wanted_history: the number of most recent values to be used for the prediction, important for the input style of the pred model
            interval_num: the number of bs values to be predicted
            models: the bag of model to do the predicting
        
        Returns:
            Predictions: 1D array of interval_num many predicted values
            indic_data: the individual predictions of the models to be further used in indication analysis
        """
        predictions = []
        indic_data = [[]] * len(models)

        # modify the simple array into scaled pd dataframe
        past_values = np.array(past_values)  # Should already be 2D from app.py
        past_values = Standard_Vars.sc.transform(past_values)  # scaler must handle 2 features

        # get only what is required for the model
        X_test = past_values[len(past_values) - wanted_history :]

        # configure X_test to fit the model
        # TODO: what does this do and is it necessary
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (1, Standard_Vars.REG_SHAPE, Standard_Vars.INPUT_DIM))

        # make prediction
        mean_pred, individual_preds = Pred_Tools.many_model_predict(X_test, models = models)
        predictions.append(mean_pred)
        
        # prepare indicator values
        indic_data = Standard_Vars.sc.inverse_transform(individual_preds)

        # append the list
        np_predictions = np.array([mean_pred])             # shape (2,)
        np_predictions = np_predictions.reshape(1, 1, -1)      # shape (1, 1, 2)
        X_test = np.append(X_test, np_predictions, axis=1) # shape becomes (1, 13, 2)

        # repeats desired interval number - 1 times from here
        # get list ready for fiting model
        for i in range(interval_num - 1):
            X_test = X_test[:, 1:, :]

            # make prediction
            mean_pred, individual_preds = Pred_Tools.many_model_predict(X_test, models = models)
            predictions.append(mean_pred)
            
            # prepare indicator values
            indic_data = Standard_Vars.sc.inverse_transform(individual_preds)

            # append the list
            np_predictions = np.array([mean_pred])             # shape (2,)
            np_predictions = np_predictions.reshape(1, 1, -1)      # shape (1, 1, 2)
            X_test = np.append(X_test, np_predictions, axis=1) # shape becomes (1, 13, 2)

        predictions = Standard_Vars.sc.inverse_transform(predictions)

        return predictions, indic_data

class Model_Assessment():

    def indicator_recognizer(values, score_list = None):
        """
        Assesses the indicator values, ready to be sent

        Args: 
            values: the set of all prediction values from all models in order in 4D array form [[[[]],[[]]],[[[]]...]]
            score_list: a 2D list [[a,b,c], ...] with scores of the models the predicted value belongs to
        
        Returns:
            Three digit indicator value with (extreme value indic, plateau indic, trend change indic) abc based on the scores of models good in that field
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
    def jsonBuilder(values, last_dexcom_instance, username, indicators):
        """
        Creates the json dictionary format
        Also updates the SQL database as it is the best time to do so

        Args: 
            values: actual bs values with the mean bs values at the end
            last_dexcom_instance: the last Dexcom value with dexcom trend info
            username: username to be able to update SQL
            indicators: the indicator values to be sent
        
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
