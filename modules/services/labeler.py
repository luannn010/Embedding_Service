import openai
import json
from modules.config import Config
from modules.template import EmbeddingJSONTemplate


class Labeler:
    def __init__(self, label_model=None):
        """
        Initializes the labeler with the specified model and host.
        Args:
            label_model (str): OpenAI model for generating labels.
        """
        self.label_model = label_model or Config.LABEL_MODEL
        openai.api_key = Config.OPENAI_API_KEY
        print("OpenAI API key set successfully.")

    def load_text_file(self, file_path):
        """
        Loads a text file and returns a list of lines.
        Args:
            file_path (str): Path to the text file.
        Returns:
            list: A list of text lines.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while reading the file: {e}")

    def create_definition(self, file_path):
        """
        Generates a definition for embedding-based collections using OpenAI's API
        and the content from the specified file.
        
        Args:
            file_path (str): Path to the text file containing input data.
        
        Returns:
            dict: A structured dictionary representing the definition.
        """
        # Load the text content from the file
        texts = self.load_text_file(file_path)

        # Combine the text lines into a single string for the prompt
        input_text = "\n".join(texts)

        try:
            # Construct the prompt for the model
            prompt = (
                f"Based on the following text content, create a structured definition for an embedding-based "
                f"collection. Include fields: collection_name, partition_name, description, dimension, and metric_type. "
                f"Ensure the response is concise and follows JSON-like formatting.\n\n"
                f"{input_text}\n\n"
                f"Example output:\n"
                f"{{\n"
                f"  \"collection_name\": \"ExampleCollection\",\n"
                f"  \"partition_name\": \"ExamplePartition\",\n"
                f"  \"description\": \"Detailed description of the collection.\",\n"
                f"  \"dimension\": 128,\n"
                f"  \"metric_type\": \"cosine\"\n"
                f"}}"
            )

            # Call the OpenAI Chat Completion API
            response = openai.ChatCompletion.create(
                model=self.label_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for creating structured definitions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            print("Connection to OpenAI API successful.")

            # Parse the generated definition as JSON
            generated_definition_str = response['choices'][0]['message']['content'].strip()
            return generated_definition_str

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse the generated definition as JSON: {e}")
        except Exception as e:
            raise Exception(f"Failed to generate definition using the model: {e}")