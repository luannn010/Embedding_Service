import openai
from modules.config import Config

# Ensure all required configurations are set
Config.validate()


class EmbeddingGenerator:
    def __init__(self, embedding_model=None):
        """
        Initializes the embedding generator with the specified models.
        Args:
            model (str): OpenAI model for generating embeddings.
            label_model (str): OpenAI model for generating labels.
        """
        self.embedding_model = embedding_model or Config.DEFAULT_MODEL
        
        openai.api_key = Config.OPENAI_API_KEY
        print("OpenAI API key set successfully for embedding generation.")

    def get_embedding(self, text):
        """
        Fetches the embedding for the given text using OpenAI's API.
        Args:
            text (str): The text to embed.
            model (str): The embedding model to use. Defaults to self.model.
        Returns:
            list: The embedding vector.
        """
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=[text]
            )
            print("Connection to OpenAI API successful for embedding generation.")
            embedding = response['data'][0]['embedding']
            print("Embedding generation completed successfully.")
            return embedding
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            raise