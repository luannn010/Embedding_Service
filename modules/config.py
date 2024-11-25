import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for embedding generator."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL = "text-embedding-3-small"
    OUTPUT_DIR = "embed"
    STREAM_SERVICE_HOST = os.getenv("STREAM_SERVICE_HOST", "localhost")
    STREAM_SERVICE_PORT = int(os.getenv("STREAM_SERVICE_PORT", 50051))
    OUTPUT_MODE = os.getenv("OUTPUT_MODE", "file")  # Options: "file" or "stream"
    LABEL_MODEL = os.getenv("LABEL_MODEL", "gpt-3.5-turbo")

    @staticmethod
    def validate():
        """Validates that all required configurations are set."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Ensure it is defined in the .env file.")
        if Config.OUTPUT_MODE not in ["file", "stream"]:
            raise ValueError("OUTPUT_MODE must be either 'file' or 'stream'.")
