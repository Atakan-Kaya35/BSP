import json
import traceback
import zipfile
import logging
from pydexcom import Dexcom
from bsp_util import Pred_Tools, Model_Assessment, Communication, Standard_Vars
from bsp_cloud_lib import Cloud_Storage
import numpy as np
from config import Config
import onnxruntime as ort

# Initialize global state
Standard_Vars.initialize()

def lambda_handler(event, context):
    """
    Lambda handler for running inference using Dexcom API and user-specific model downloaded from S3.

    Expects:
    - JSON body with 'username' and 'password' as strings

    Returns:
    - JSON result with prediction and indicators, see Communications.jsonBuilder() for more info
    """
    
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

        # Step 1: Get Dexcom readings
        dexcom = Dexcom(username, password, ous=True)
        glucose_readings = dexcom.get_glucose_readings(max_count = 12)
        prevs = np.array([
            [float(glucose_readings[i].value), 
            # Original: tod exemption try
            np.sin(2 * np.pi * (glucose_readings[i].datetime.hour * 60 + glucose_readings[i].datetime.minute) / 1440),
            np.cos(2 * np.pi * (glucose_readings[i].datetime.hour * 60 + glucose_readings[i].datetime.minute) / 1440)
            ]
            for i in range(Standard_Vars.REG_SHAPE)
        ])[::-1] 
        Standard_Vars.current_time = glucose_readings[0].datetime.hour * 60 + glucose_readings[0].datetime.minute
        
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
        model_paths = [Config.TMP_DIR / f"{username}_{i}.onnx" for i in range(1, metadata[0]+1)]
        # if not all the models exist in the desired path
        if not all(p.exists() for p in model_paths):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for i in range(1, metadata[0]+1):
                    zip_ref.extract(f"{username}_{i}.onnx", path=str(Config.TMP_DIR))
                    zip_ref.extract(f"{username}_{i}_scores.json", path=str(Config.TMP_DIR))

        # Step 4: Load model and scores
        personal_model = []
        for i in range(1, metadata[0] + 1):
            onnx_path = Config.TMP_DIR / f"{username}_{i}.onnx"
            sess = ort.InferenceSession(str(onnx_path))
            personal_model.append(sess)
            
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

        # append the new predictions
        # Convert prevs to a list so we can append
        prevs = np.concatenate([prevs, nexts], axis=0)

        # Step 6: Generate indicators
        indicators = Model_Assessment.indicator_recognizer(indic_data, indic_score_list)

        # Step 7: Return result
        response = Communication.jsonBuilder(prevs, glucose_readings[0], indicators)
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

    #finally:
        # CAUTION: commands can be added to erase the artifacts downloaded in oder to save costs
        # this is not best practice right now due to the hot start nature of AWS optimization
        # Delete the local files
        """ 
        try:
            os.remove(f"/tmp/{username}.h5")
            os.remove(f"/tmp/{username}_scores.json")
        except Exception:
            pass
        """

if Config.IS_LOCAL:
    lambda_handler(0,0)
