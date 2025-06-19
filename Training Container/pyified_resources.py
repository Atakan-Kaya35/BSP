from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class Standard_Vars:
    FIVE_MIN_INTERVAL = 12
    REG_SHAPE = 12
    sc = None
    mydb = None

    @classmethod
    def initialize(cls, seq_len):
        cls.FIVE_MIN_INTERVAL = seq_len
        cls.REG_SHAPE = seq_len
        
        # Initialize scaler for both glucose (40-400) and time (0-1)
        # 39 to prevent division by 0 when there are 40 the real value vecomes 0 which causes error, one below 40 makes everything above 0
        values = np.array([
            [39, 0],   # Min glucose, min time
            [400, 1]    # Max glucose, max time
        ])
        cls.sc = MinMaxScaler(feature_range=(0, 1))
        cls.sc.fit(values)
        
        evaluation_datasets = []
        for file in [
            "BSP_Extreme_Value_Evaluator_Models.csv",
            "BSP_Plateau_Evaluator_Models.csv",
            "BSP_General_Trend_Change_Evaluator_Models.csv"
        ]:
            data = pd.read_csv(file).values  # shape: (n_samples, 1) or (n_samples,)

            # Add [0.5, value] rows
            transformed = [[[value, 0.5] for value in row] for row in data]
        
            evaluation_datasets.append(transformed)


        
        cls.evaluation_datasets = evaluation_datasets


