import unittest
import unittest.mock as mock
import pandas as pd
import awswrangler as wr
from real_estate_model.flows.model_batch_prediction import generate_prediction, upload_prediction_to_s3

class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = X.shape[0]
        return [self.value] * n

class TestBatchPrediction(unittest.TestCase):        
    def test_generate_prediction(self):
        # test generate_prediction function
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        model = ModelMock(10)
        result = generate_prediction.fn(data, model)
        expected_result = [10, 10, 10]
        self.assertEqual(result, expected_result)

    def test_upload_prediction_to_s3(self):
        # test upload_prediction_to_s3 function
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        out_filepath = "s3://bucket/predictions.csv"
        with mock.patch.object(wr.s3, "to_csv") as mock_wr_s3_to_csv:
            result = upload_prediction_to_s3.fn(data, out_filepath)
            mock_wr_s3_to_csv.assert_called_once_with(data, out_filepath, index=False)
        self.assertTrue(result)