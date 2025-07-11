import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import numpy as np

""" class Models():
    REGRESSOR1 = load_model("67_acc_model(12,50,24,10).h5")
    MODEL_MIX = [REGRESSOR1] """

class Standard_Vars:
    FIVE_MIN_INTERVAL = 12
    REG_SHAPE = 12
    # Original is 2 as it is bsv, tod exemption
    INPUT_DIM = 3
    sc = None
    sc_sugar = None
    mydb = None
    onnx_session = None
    current_time = 0    

    @classmethod
    def initialize(cls):
        # 39 so the min value of 40 does not lead to 0 which causes copmlications
        values = np.array([
            [39, -1, -1], [400, 1, 1]
            # Original: tod exemption
            #[39, 0], [400, 1]
            ])
        cls.sc = MinMaxScaler(feature_range=(0, 1))
        cls.sc.fit(values)
        
        values = np.array([
            [39], [400]
            # Original: tod exemption
            #[39, 0], [400, 1]
            ])
        cls.sc_sugar = MinMaxScaler(feature_range=(0, 1))
        cls.sc_sugar.fit(values)
        
        """time_values = np.array([
            [-1], [1]
        ])
        cls.sc_time = MinMaxScaler(feature_range=(0, 1))
        cls.sc_time.fit(time_values) """

        # connect to db
        # TODO: sql db for user tracking
        
    @classmethod
    def load_model(cls, model_path="model.onnx"):
        cls.onnx_session = ort.InferenceSession(model_path)


