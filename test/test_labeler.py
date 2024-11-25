from modules.services.labeler import Labeler
import unittest
from unittest.mock import patch, MagicMock



class TestLabeler(unittest.TestCase):
    def setUp(self):
        """Set up a Labeler instance before each test."""
        self.labeler = Labeler(label_model="gpt-4")
        self.test_file_path = "test_file.txt"

        # Sample text content for mocking file reads
        self.sample_text = "This is a sample text file used for testing."

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_text_file(self, mock_open):
        """Test loading text from a file."""
        mock_open.return_value.__enter__.return_value = self.sample_text.splitlines()
        
        result = self.labeler.load_text_file(self.test_file_path)
        expected_result = ["This is a sample text file used for testing."]
        self.assertEqual(result, expected_result)

    @patch("openai.ChatCompletion.create")
    @patch("modules.services.labeler.Labeler.load_text_file")
    def test_create_definition(self, mock_load_text_file, mock_openai_create):
        """Test creating a definition using OpenAI's API."""
        # Mock the content returned by load_text_file
        mock_load_text_file.return_value = [self.sample_text]

        # Mock the OpenAI ChatCompletion API response
        mock_openai_create.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "{\n"
                            "  \"collection_name\": \"SampleCollection\",\n"
                            "  \"partition_name\": \"SamplePartition\",\n"
                            "  \"description\": \"A test collection of sample data.\",\n"
                            "  \"dimension\": 128,\n"
                            "  \"metric_type\": \"cosine\"\n"
                            "}"
                        )
                    }
                }
            ]
        }

        # Call the create_definition method
        result = self.labeler.create_definition(self.test_file_path)

        # Expected result from the mocked API response
        expected_result = (
            "{\n"
            "  \"collection_name\": \"SampleCollection\",\n"
            "  \"partition_name\": \"SamplePartition\",\n"
            "  \"description\": \"A test collection of sample data.\",\n"
            "  \"dimension\": 128,\n"
            "  \"metric_type\": \"cosine\"\n"
            "}"
        )

        self.assertEqual(result, expected_result)

    def test_create_definition_file_not_found(self):
        """Test create_definition raises FileNotFoundError for a missing file."""
        with self.assertRaises(FileNotFoundError):
            self.labeler.create_definition("nonexistent_file.txt")


# if __name__ == "__main__":
#     unittest.main()
