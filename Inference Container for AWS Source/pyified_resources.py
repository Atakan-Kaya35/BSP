from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Models():
    REGRESSOR1 = load_model("67_acc_model(12,50,24,10).h5")
    MODEL_MIX = [REGRESSOR1]

class Standard_Vars:
    FIVE_MIN_INTERVAL = 12
    REG_SHAPE = 12
    # Original is 2 as it is bsv, tod exemption
    INPUT_DIM = 1
    sc = None
    mydb = None

    @classmethod
    def initialize(cls):
        # 39 so the min value of 40 does not lead to 0 which causes copmlications
        values = np.array([
            [39], [400]
            # Original: tod exemption
            #[39, 0], [400, 1]
            ])
        cls.sc = MinMaxScaler(feature_range=(0, 1))
        cls.sc.fit(values)

        # connect to db
        # TODO: connection not made since its auto ending by the server-side abrupts operations
        """ try:
            cls.mydb = mysql.connector.connect(
                host="***REDACTED***",
                user="***REDACTED***",
                password="***REDACTED-ROTATED-CREDENTIAL***",
                database="***REDACTED***",
                connect_timeout=6000
            )
            
            cls.mycursor = cls.mydb.cursor(buffered=True)

            print("Connected successfully!")

        except Exception as e:
            print(f"Error connecting to MySQL: {e}") """


