import unittest
from src import feature_flags

class TestFeatureFlags(unittest.TestCase):

    def test_example_feature_1_disabled(self):
        self.assertFalse(feature_flags.is_feature_enabled('example_feature_1'))

    def test_example_feature_2_enabled(self):
        self.assertTrue(feature_flags.is_feature_enabled('example_feature_2'))

if __name__ == '__main__':
    unittest.main()
