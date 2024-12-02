import os
from modules.proto.embedding.embedding_buffer_pb2 import EmbeddingResponse
from modules.services import Labeler, EmbeddingGenerator
from modules.template import EmbeddingJSONTemplate

class EmbeddingService:
    def __init__(self):
        self.labeler = Labeler()
        self.embedder = EmbeddingGenerator()
        self.embedded_dir = "embed/embedded"
        os.makedirs(self.embedded_dir, exist_ok=True)  # Ensure the directory exists

    def _get_embedded_file_path(self, file_name):
        """Construct the path to the embedded file based on the file name."""
        name, _ = os.path.splitext(file_name)  # Extract the base name without extension
        return os.path.join(self.embedded_dir, f"{name}_embedded.json")

    def _read_embedded_file(self, file_path):
        """Read the content of an existing embedded file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _save_embedded_file(self, file_path, content):
        """Save the content to an embedded file."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Embedded file saved at: {file_path}")

    def StreamEmbedding(self, request, context):
        try:
            # Extract file name from the request
            file_name = request.file_name
            if not file_name:
                raise ValueError("File name is missing in the request.")

            # Determine the path to the embedded file
            embedded_file_path = self._get_embedded_file_path(file_name)

            # Check if the embedded file already exists
            if os.path.exists(embedded_file_path):
                print(f"Embedded file '{embedded_file_path}' already exists. Using the cached file.")
                cached_content = self._read_embedded_file(embedded_file_path)
                yield EmbeddingResponse(json_stream=cached_content)
                return

            # Read file content from the stream
            file_content = request.file_stream
            if not file_content:
                raise ValueError("File stream is empty.")

            # Use the Labeler to create a definition from the file content
            definition_json = self.labeler.create_definition_from_content(file_content)
            print("Definition JSON:", definition_json)

            # Use the EmbeddingGenerator to get embeddings for the text
            embedding_vector = self.embedder.get_embedding(definition_json)
            print("Embedding Vector:", embedding_vector)
            print("Type of Embedding Vector:", type(embedding_vector))

            # Ensure embedding_vector is a list of floats
            if not isinstance(embedding_vector, list) or not all(isinstance(x, float) for x in embedding_vector):
                raise ValueError("Embedding vector must be a list of floats.")

            # Create a JSON stream using the EmbeddingJSONTemplate
            stream = EmbeddingJSONTemplate(definition_json, embedding_vector)
            json_content = stream.to_json()

            # Save the JSON content to the embedded file
            self._save_embedded_file(embedded_file_path, json_content)

            # Respond with the generated JSON stream
            yield EmbeddingResponse(json_stream=json_content)

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            raise
