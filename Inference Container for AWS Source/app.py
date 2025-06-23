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
    
    # An options tyoe http call in a rest api requires special treatment, 
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
        # LOCAL: manuelly complete this
        """username = "atakanka350@gmail.com"
        password = "***REDACTED-ROTATED-CREDENTIAL***"
        """ 
        body = json.loads(event['body'])
        username = body.get('username')
        password = body.get('password')
        # /LOCAL
        
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

        # TODO: Unnecessary for the AWS Lambda implementation as the data is just got anyway, no hist to compare with
        # Step 2: Handle skipped values
        """ skipped = 0
        if past[-6:] != prevs[-7:-1]:
            if past[-6:] != prevs[-8:-2]:
                if past[-6:] != prevs[-9:-3]:
                    if past[-6:] != prevs[-10:-4]:
                        pass
                    else:
                        skipped = 3
                else:
                    skipped = 2
            else:
                skipped = 1

        # was_predicted is a list of the previous predictions, this for loop puts an empty list in the place of 
        # the prediction that should have heppened but did not due to the lag in the dexcom system
        for _ in range(skipped):
            for i in range(1, Standard_Vars.REG_SHAPE + 1):
                was_predicted[i - 1] = was_predicted[i]
            was_predicted[-1] = []
 """
        # Step 3: Download and extract zip
        # LOCAL: test the paths are made from / to \\ , comment out the s3 get method as it is not needed
        # if you need to test a new set of models manuelly download from s3 etc and place inside the tmp folder as zip file
        # (there is another change at the download_from_s3 function about the making sure the tmp directory exists)
        zip_file = f"/tmp/{username}_data.zip"
        if not os.path.exists(zip_file):
            Cloud_Storage.download_from_s3(f"{username}/{username}_data.zip", zip_file)
            
        # first read the metadata file 
        metadata_file = f'{username}_metadata.json'
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extract(metadata_file, '/tmp/')

        with open(f"/tmp/{metadata_file}", 'r') as file:
            metadata = list(json.load(file).values())

        # metadata[0] contains how many models there are
        model_paths = [f"/tmp/{username}_{i}.h5" for i in range(1, metadata[0]+1)]
        if not all(os.path.exists(p) for p in model_paths):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for i in range(1, metadata[0]+1):
                    zip_ref.extract(f"{username}_{i}.h5", '/tmp/')
                    zip_ref.extract(f"{username}_{i}_scores.json", '/tmp/')

        indic_score_list = []

        # Step 4: Load model and scores
        personal_model = [load_model(f"/tmp/{username}_{i}.h5") for i in range(1, metadata[0]+1)]
        for i in range(1, metadata[0]+1):
            with open(f"/tmp/{username}_{i}_scores.json", 'r') as file:
                indic_score_list.append(list(json.load(file).values()))
        # /LOCAL

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

#LOCAL: DO NOT call the function otherwise
#lambda_handler(0,0)
# /LOCAL