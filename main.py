import grpc
from concurrent import futures
import time
from modules.proto.embedding.embedding_buffer_pb2_grpc import add_EmbeddingServiceServicer_to_server
from modules.services.embedding_service import EmbeddingService  # Import the class

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_service = EmbeddingService()  # Instantiate the service
    add_EmbeddingServiceServicer_to_server(embedding_service, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started, listening on port 50051.")
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
