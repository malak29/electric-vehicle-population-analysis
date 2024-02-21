import unittest
import pandas as pd
from your_utility_file import load_and_duplicate_data, replace_nans, fill_missing_values, drop_columns, encode_labels
# Replace 'your_utility_file' with the actual name of your Python file containing the utility functions

class TestUtilityFunctions(unittest.TestCase):

    def setUp(self):
        # Example setup creating a DataFrame to use in tests
        self.example_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['a', 'b', 'c', np.nan],
            'C': [np.nan, 'y', 'z', 'x']
        })

    def test_load_and_duplicate_data(self):
        # This test would need to mock or simulate the pd.read_csv function or ensure a test file exists at a known location
        pass  # Implement test logic here

    def test_replace_nans(self):
        # Test replacing NaNs in column 'A' with a specific value, assuming 'clean_column_replace_nan_with_1' replaces NaN with 1
        modified_df = replace_nans(self.example_df.copy(), ['A'])
        self.assertFalse(modified_df['A'].isnull().any())

    def test_fill_missing_values(self):
        # Test filling missing values in column 'B' with 'unknown'
        modified_df = fill_missing_values(self.example_df.copy(), {'B': 'unknown'})
        self.assertFalse(modified_df['B'].isnull().any())

    def test_drop_columns(self):
        # Test dropping column 'C'
        modified_df = drop_columns(self.example_df.copy(), ['C'])
        self.assertNotIn('C', modified_df.columns)

    def test_encode_labels(self):
        # Assuming 'B' is a categorical column that needs encoding
        modified_df = encode_labels(self.example_df.copy(), ['B'])
        # Check if column 'B' has been converted to numeric type
        self.assertTrue(pd.api.types.is_numeric_dtype(modified_df['B']))

    # Add more tests for other functions...

if __name__ == '__main__':
    unittest.main()