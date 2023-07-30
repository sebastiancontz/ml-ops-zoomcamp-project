import unittest
from unittest.mock import patch, Mock
from mlflow.exceptions import MlflowException
from real_estate_model.api import create_app

class MockModel():
    def __init__(self):
        self.metadata = Mock()
        self.metadata.run_id = "1234"
        self.metadata.model_uuid = "5678"

    def predict(self, X):
        return [10] * X.shape[0]

class ApiIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_valid_input(self):
        input_data = {
            "house_age": 10,
            "distance_to_the_nearest_MRT_station": 200,
            "number_of_convenience_stores": 5,
            "latitude": 25.12345,
            "longitude": 121.56789
        }

        with patch('mlflow.pyfunc.load_model') as mock_pyfunc_load_model:
            mock_pyfunc_load_model.return_value = MockModel()
            response = self.client.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(response.json["input_data"], input_data)
        self.assertEqual(response.json["prediction"], 10)
        self.assertEqual(response.json["model_metadata"]["run_id"], "1234")
        self.assertEqual(response.json["model_metadata"]["model_uuid"], "5678")

    def test_missing_parameters(self):
        input_data = {
            "house_age": 10,
            "distance_to_the_nearest_MRT_station": 200,
            "number_of_convenience_stores": 5,
            "latitude": 25.12345
        }

        with patch('mlflow.pyfunc.load_model') as mock_pyfunc_load_model:
            mock_pyfunc_load_model.return_value = MockModel()
            response = self.client.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Missing parameters: longitude")

    def test_model_not_trained(self):
        def mock_load_model(model_uri):
            raise MlflowException(message="Model not found")

        input_data = {
            "house_age": 10,
            "distance_to_the_nearest_MRT_station": 200,
            "number_of_convenience_stores": 5,
            "latitude": 25.12345
        }

        with patch('mlflow.pyfunc.load_model') as mock_pyfunc_load_model:
            mock_pyfunc_load_model.side_effect = mock_load_model
            response = self.client.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["error"], "Model not trained yet, please train a model within prefect UI or calling /trigger-training endpoint")

    def test_trigger_batch_prediction_with_valid_s3_file_path(self):
        s3_file_path = "s3://bucket_name/file.csv"
        with patch("real_estate_model.utils.functions.get_prefect_flow_id") as mock_get_flow_id, \
            patch("requests.post") as mock_post_request:
            mock_get_flow_id.return_value = "flow-id"
            mock_post_request.return_value.status_code = 201
            mock_post_request.return_value.json.return_value = {
                "id": "flow-run-id",
                "name": "flow-run-name"
            }

            input_data = {"s3_file_path": s3_file_path}
            response = self.client.post("/trigger-batch-prediction", json=input_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("flow_run_id", response.json)
        self.assertIn("flow_run_name", response.json)

    def test_trigger_batch_prediction_with_missing_s3_file_path(self):
        response = self.client.post("/trigger-batch-prediction", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertEqual(response.json["error"], "Missing parameters: s3_file_path")

    def test_trigger_batch_prediction_with_prefect_error(self):
        s3_file_path = "s3://bucket_name/file.csv"
        with patch("real_estate_model.utils.functions.get_prefect_flow_id") as mock_get_flow_id, \
             patch("requests.post") as mock_post_request:
            mock_get_flow_id.return_value = "flow-id"
            mock_post_request.return_value.status_code = 500

            response = self.client.post(
                "/trigger-batch-prediction",
                json={"s3_file_path": s3_file_path}
            )

            self.assertEqual(response.status_code, 500)
            self.assertIn("error", response.json)
            self.assertEqual(response.json["error"], "Something went wrong")

if __name__ == '__main__':
    unittest.main()
