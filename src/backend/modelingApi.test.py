import unittest
import json
from app import app  # Import the Flask application instance from your app.py file

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        # Propagate the exceptions to the test client
        self.app.testing = True

    def test_load_data(self):
        # Test the /load_data endpoint
        response = self.app.post('/load_data', data=json.dumps({'file_path': 'your_file_path.csv'}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Data loaded successfully', response.json['message'])

    def test_clean_data(self):
        # Test the /clean_data endpoint
        response = self.app.post('/clean_data', data=json.dumps({'df_name': 'your_df_name', 'nans_columns': ['A', 'B']}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Data cleaned successfully', response.json['message'])

    def test_execute_model_decision_tree(self):
        # Test the /execute_model endpoint for the decision tree model
        data = {
            'model_name': 'decision_tree',
            'X': [{'feature1': 1, 'feature2': 2}, {'feature1': 3, 'feature2': 4}],  # Example feature data
            'y': [0, 1]  # Example target variable
        }
        response = self.app.post('/execute_model', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Model executed successfully', response.json['message'])
        # Add more assertions here based on the expected result format

    # Add more test methods for other models and endpoints...

if __name__ == '__main__':
    unittest.main()
