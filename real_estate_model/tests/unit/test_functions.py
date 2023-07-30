import unittest
from unittest.mock import patch
import pandas as pd
from pandas.core.indexes.base import Index
from flask import Flask, request
from real_estate_model.utils import functions


class TestProcessCsv(unittest.TestCase):

    def test_process_csv_no_predict(self):
        dummy_df = pd.DataFrame(
            data={}, 
            columns=["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", 
                    "latitude", "longitude", "house_price_of_unit_area"]
        )
        expected = Index(
            ["house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", 
            "latitude", "longitude", "house_price_of_unit_area"], dtype="object")
        
        processed_df = functions.process_csv(dummy_df, is_prediction=False)

        assert processed_df.columns.equals(expected) == True

    def test_process_csv_predict(self):
        dummy_df = pd.DataFrame(
            data={}, 
            columns=["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", 
                    "latitude", "longitude"]
        )
        expected = Index(
            ["house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", 
            "latitude", "longitude"], dtype="object")
        
        processed_df = functions.process_csv(dummy_df, is_prediction=True)

        assert processed_df.columns.equals(expected) == True

class TestValidateHeaders(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)

    def test_validate_headers_with_missing_parameters(self):
        # case 1: verify that the function correctly detects a missing expected parameter.
        with self.app.test_request_context('/some_endpoint', json={'param1': 'value1', 'param3': 'value3'}):
            missing, parameters = functions.validate_headers(request, expected=['param1', 'param2', 'param3'])
            assert missing is True
            assert parameters == 'param2'

    def test_validate_headers_with_all_parameters(self):
        # case 2: verify that the function returns no missing parameters when all expected parameters are present.
        with self.app.test_request_context('/some_endpoint', json={'param1': 'value1', 'param2': 'value2', 'param3': 'value3'}):
            missing, parameters = functions.validate_headers(request, expected=['param1', 'param2', 'param3'])
            assert missing is False
            assert parameters is None

    def test_validate_headers_with_empty_json(self):
        # case 3: verify that the function correctly detects all expected parameters as missing when an empty JSON is provided.
        with self.app.test_request_context('/some_endpoint', json={}):
            expected = ['param1', 'param2', 'param3']
            missing, parameters = functions.validate_headers(request, expected=expected)
            parameters = parameters.split(', ')
            assert missing is True
            assert set(parameters) == set(expected)


class TestGetPrefectFlowID(unittest.TestCase):
    @patch("os.getenv", return_value="http://example.com/prefect/api")
    @patch("requests.get")
    def test_get_prefect_flow_id(self, mock_requests_get, mock_os_getenv):
        expected_flow_name = "my_flow"
        expected_flow_id = "123456"
        mock_response = mock_requests_get.return_value
        mock_response.json.return_value = {"id": expected_flow_id}

        flow_id = functions.get_prefect_flow_id(expected_flow_name)

        mock_os_getenv.assert_called_once_with("PREFECT_API_URL")
        mock_requests_get.assert_called_once_with("http://example.com/prefect/api/deployments/name/my_flow/my_flow")

        self.assertEqual(flow_id, expected_flow_id)