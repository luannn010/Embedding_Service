import os
import grpc
import modules.proto.embedding.embedding_buffer_pb2 as embedding_pb2
import modules.proto.embedding.embedding_buffer_pb2_grpc as embedding_pb2_grpc

def get_file_stream(file_path):
    """Read the file and return its content as bytes."""
    with open(file_path, "rb") as file:
        return file.read()

def get_json_stream(file_path, file_stream):
    """Send the file stream and file name via gRPC."""
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
            file_name = os.path.basename(file_path)
            request = embedding_pb2.EmbeddingRequest(file_name=file_name, file_stream=file_stream)
            response_iterator = stub.StreamEmbedding(request)

            for response in response_iterator:
                print("Received response from server.")
                print("JSON stream:", response.json_stream)

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    test_file_path = "embed/stories/small_village.txt"
    file_stream = get_file_stream(test_file_path)
    print("Streaming JSON response:")
    get_json_stream(test_file_path, file_stream)
