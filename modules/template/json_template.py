import json


class EmbeddingJSONTemplate:
    def __init__(self, definition: str, embedding: list):
        """
        Initializes the JSON template with a given definition and embeddings.

        Args:
            definition (str): A JSON string containing collection_name, partition_name,
                              description, dimension, and metric_type.
            embedding (list): A list of embeddings.
        """
        self.template = {
            "definition": self._parse_and_validate_definition(definition),
            "embeddings": self._flatten_embedding(embedding)
        }
        # Automatically generate JSON
        self.to_json()

    def _parse_and_validate_definition(self, definition: str):
        """
        Parses and validates the definition JSON string.

        Args:
            definition (str): A JSON string containing collection_name, partition_name,
                              description, dimension, and metric_type.

        Returns:
            dict: The parsed and validated definition dictionary.
        """
        try:
            definition_dict = json.loads(definition)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse the definition as JSON: {e}")

        # Validate required fields in the definition
        required_fields = {"collection_name", "partition_name", "description", "dimension", "metric_type"}
        if not required_fields.issubset(definition_dict.keys()):
            raise ValueError(f"Missing required fields in the definition. Required fields are: {required_fields}")

        return definition_dict

    def _flatten_embedding(self, embedding: list):
        """
        Flattens the embedding if it's a list of lists.

        Args:
            embedding (list): A list of embeddings.

        Returns:
            list: A flattened list of embeddings.
        """
        if any(isinstance(i, list) for i in embedding):
            return [item for sublist in embedding for item in sublist]
        return embedding

    def to_json(self):
        """
        Converts the template into a JSON string.

        Returns:
            str: JSON-formatted string of the template.
        """
        return json.dumps(self.template, indent=4)

    def save_to_file(self, file_path="output.json"):
        """
        Saves the JSON template to a file.

        Args:
            file_path (str): Path to save the JSON file.
        """
        with open(file_path, 'w') as file:
            json.dump(self.template, file, indent=4)
        print(f"JSON saved to {file_path}")


# # Example usage
# if __name__ == "__main__":
#     definition_json = '''
#     {
#         "collection_name": "example_collection",
#         "partition_name": "example_partition",
#         "description": "An example definition.",
#         "dimension": 128,
#         "metric_type": "L2"
#     }
#     '''
#     embedding_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

#     # Automatically generate the JSON output
#     template = EmbeddingJSONTemplate(definition_json, embedding_list)

#     # Access the JSON output if needed
#     json_output = template.json_output
