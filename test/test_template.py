import unittest
import json
from modules.template.json_template import EmbeddingJSONTemplate  # Adjust the import based on your file structure


class TestEmbeddingJSONTemplate(unittest.TestCase):
    def setUp(self):
        """
        Set up test data for the tests.
        """
        self.valid_definition = '''
        {
            "collection_name": "test_collection",
            "partition_name": "test_partition",
            "description": "This is a test collection.",
            "dimension": 128,
            "metric_type": "L2"
        }
        '''
        self.invalid_definition = '''
        {
            "collection_name": "test_collection",
            "description": "This is a test collection.",
            "dimension": 128
        }
        '''
        self.valid_embedding = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.single_list_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_valid_template(self):
        """
        Test the class with valid definition and embedding.
        """
        template = EmbeddingJSONTemplate(self.valid_definition, self.valid_embedding)
        result = json.loads(template.to_json())
        self.assertIn("definition", result)
        self.assertIn("embeddings", result)
        self.assertEqual(result["definition"]["collection_name"], "test_collection")
        self.assertEqual(len(result["embeddings"]), 6)  # Flattened list

    def test_invalid_definition(self):
        """
        Test the class with an invalid definition.
        """
        with self.assertRaises(ValueError) as context:
            EmbeddingJSONTemplate(self.invalid_definition, self.valid_embedding)
        self.assertIn("Missing required fields", str(context.exception))

    def test_flatten_embedding(self):
        """
        Test that embeddings are correctly flattened.
        """
        template = EmbeddingJSONTemplate(self.valid_definition, self.valid_embedding)
        result = json.loads(template.to_json())
        self.assertEqual(result["embeddings"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_single_list_embedding(self):
        """
        Test with already flat embeddings.
        """
        template = EmbeddingJSONTemplate(self.valid_definition, self.single_list_embedding)
        result = json.loads(template.to_json())
        self.assertEqual(result["embeddings"], self.single_list_embedding)

    def test_save_to_file(self):
        """
        Test the save_to_file method.
        """
        template = EmbeddingJSONTemplate(self.valid_definition, self.valid_embedding)
        file_path = "test_output.json"
        template.save_to_file(file_path)
        with open(file_path, "r") as file:
            saved_data = json.load(file)
        self.assertIn("definition", saved_data)
        self.assertIn("embeddings", saved_data)


if __name__ == "__main__":
    unittest.main()
