import json
import os
import traceback
import zipfile
import logging
from keras.models import load_model
from pydexcom import Dexcom
from bsp_util import Pred_Tools, Model_Assessment, Communication, Standard_Vars
from bsp_cloud_lib import Cloud_Storage
import numpy as np
from config import Config
from pathlib import Path

# Initialize global state
past = [0.0] * 7
was_predicted = [[] for _ in range(Standard_Vars.REG_SHAPE + 1)]
follower = 0
Standard_Vars.initialize()

def lambda_handler(event, context):
    """
    Lambda handler for running inference using Dexcom API and user-specific model.

    Expects:
    - JSON body with 'username' and 'password'

    Returns:
    - JSON result with prediction and indicators
    """
    
    # An options type http call in a rest api requires special treatment, 
    # if the lambda function switches to a rest api from a https api recponsider code
    """if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }"""
    
    try:
        if Config.IS_LOCAL:
            username = "atakanka350@gmail.com"
            password = "***REDACTED-ROTATED-CREDENTIAL***"
        else:
            body = json.loads(event['body'])
            username = body.get('username')
            password = body.get('password')
                
        if not username or not password:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'username' or 'password'"})
            }

        global past, was_predicted, follower

        # Step 1: Get Dexcom readings
        dexcom = Dexcom(username, password, ous=True)
        glucose_readings = dexcom.get_glucose_readings(max_count = 12)
        prevs = np.array([
            [float(glucose_readings[i].value), 
            # Original: tod exemption try
            #(glucose_readings[i].datetime.hour * 60 + glucose_readings[i].datetime.minute) / 1440
            ]
            for i in range(Standard_Vars.REG_SHAPE)
        ])[::-1] 
        print(prevs)

        # Step 3: Download and extract zip
        #whether local or in production, there must be a tmp folder in the same directory as the scrpit that contains the 
        #artifacts needed, in this case the zip file
        zip_file = Config.TMP_DIR / f"{username}_data.zip"
        if not zip_file.exists() and not Config.IS_LOCAL:
            Cloud_Storage.download_from_s3(f"{username}/{username}_data.zip", str(zip_file))

        #download json metadata file
        metadata_filename = f"{username}_metadata.json"
        metadata_path = Config.TMP_DIR / metadata_filename
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extract(metadata_filename, path=str(Config.TMP_DIR))
        #read and extract the metadata files into a list
        with open(metadata_path, 'r') as file:
            metadata = list(json.load(file).values())

        # metadata[0] contains how many models there are
        model_paths = [Config.TMP_DIR / f"{username}_{i}.h5" for i in range(1, metadata[0]+1)]
        # if not all the models exist in the desired path
        if not all(p.exists() for p in model_paths):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for i in range(1, metadata[0]+1):
                    zip_ref.extract(f"{username}_{i}.h5", path=str(Config.TMP_DIR))
                    zip_ref.extract(f"{username}_{i}_scores.json", path=str(Config.TMP_DIR))

        # Step 4: Load model and scores
        personal_model = [
            load_model(str(Config.TMP_DIR / f"{username}_{i}.h5"))
            for i in range(1, metadata[0]+1)
        ]
        #load the scores of the models into a model in order
        indic_score_list = []
        for i in range(1, metadata[0]+1):
            score_path = Config.TMP_DIR / f"{username}_{i}_scores.json"
            with open(score_path, 'r') as file:
                indic_score_list.append(list(json.load(file).values()))


        # Step 5: Run prediction
        nexts, indic_data = Pred_Tools.pred_next_arbitrary(prevs.copy(), 
                                                           Standard_Vars.FIVE_MIN_INTERVAL, 
                                                           models = personal_model)

        follower += 1
        # shift previous predictions by one
        for i in range(1, Standard_Vars.REG_SHAPE + 1):
            was_predicted[i - 1] = was_predicted[i]

        # TODO: for continuous inferences dexcom skipping BS values in sometimes sending them late was a problem
        # this is one of the codes written to remedy that situation
        # append the new predictions
        was_predicted[-1] = nexts
        past = prevs.copy()
        # Convert prevs to a list so we can append
        prevs = np.concatenate([prevs, nexts], axis=0)

        # Step 6: Generate indicators
        indicators = Model_Assessment.indicator_recognizer(indic_data, indic_score_list)

        # Step 7: Return result
        response = Communication.jsonBuilder(prevs, glucose_readings[0], username, indicators)
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(error_trace)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    finally:
        # CAUTION: the following commands are erased to save costs and since the training 
        # container is very disposable, a good and long term arhitecture should leave no
        # file behind and this is not a best practice, through it is for this use case 
        # Delete the local files
        """ 
        try:
            os.remove(f"/tmp/{username}.h5")
            os.remove(f"/tmp/{username}_scores.json")
        except Exception:
            pass
        """
""" if __name__ == "__main__":
    # Simulated Lambda event payload (matches what API Gateway sends)
    test_event = {
        "body": json.dumps({
            "username": "atakanka350@gmail.com",   # Replace with real credentials
            "password": "***REDACTED-ROTATED-CREDENTIAL***"
        })
    }

    # Dummy context (can be None unless you're using it)
    test_context = None

    # Call the Lambda handler
    response = lambda_handler(test_event, test_context)

    # Pretty-print the result
    print("Lambda test output:")
    print(json.dumps(response, indent=4)) """

if Config.IS_LOCAL:
    lambda_handler(0,0)
