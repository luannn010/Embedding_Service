import grpc
from concurrent import futures
import time
from modules.proto.embedding.embedding_buffer_pb2_grpc import EmbeddingServiceServicer, add_EmbeddingServiceServicer_to_server
from modules.proto.embedding.embedding_buffer_pb2 import EmbeddingResponse
from modules.services import Labeler, EmbeddingGenerator
from modules.template import EmbeddingJSONTemplate

class EmbeddingService(EmbeddingServiceServicer):
    def __init__(self):
        self.labeler = Labeler()
        self.embedder = EmbeddingGenerator()

    def StreamEmbedding(self, request, context):
        try:
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
            response = EmbeddingResponse(json_stream=stream.to_json())
            yield response

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EmbeddingServiceServicer_to_server(EmbeddingService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051.")
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
