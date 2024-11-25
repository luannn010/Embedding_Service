import unittest
from unittest.mock import patch, MagicMock
from modules.services.embedder import EmbeddingGenerator


class TestEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        """Set up an EmbeddingGenerator instance before each test."""
        self.embedding_generator = EmbeddingGenerator(embedding_model="text-embedding-ada-002")

    @patch("openai.Embedding.create")
    def test_get_embedding_success(self, mock_openai_create):
        """Test successful embedding generation using OpenAI's API."""
        # Mock the OpenAI API response
        mock_openai_create.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3, 0.4]}
            ]
        }

        # Sample text input
        sample_text = "This is a test input."

        # Call the get_embedding method
        result = self.embedding_generator.get_embedding(sample_text)

        # Assert that the mock response was used
        mock_openai_create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=[sample_text]
        )

        # Expected embedding result
        expected_result = [0.1, 0.2, 0.3, 0.4]

        # Validate the result
        self.assertEqual(result, expected_result)

    @patch("openai.Embedding.create")
    def test_get_embedding_api_error(self, mock_openai_create):
        """Test handling of API errors during embedding generation."""
        # Simulate an exception raised by the OpenAI API
        mock_openai_create.side_effect = Exception("API connection failed")

        # Sample text input
        sample_text = "This is a test input."

        # Assert that an exception is raised
        with self.assertRaises(Exception) as context:
            self.embedding_generator.get_embedding(sample_text)

        # Validate the exception message
        self.assertIn("API connection failed", str(context.exception))

    def test_get_embedding_invalid_text(self):
        """Test handling of invalid text input."""
        invalid_text = None  # Example of invalid input

        # Assert that an exception is raised
        with self.assertRaises(Exception):
            self.embedding_generator.get_embedding(invalid_text)


if __name__ == "__main__":
    unittest.main()
