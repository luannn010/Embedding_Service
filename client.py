import grpc
import modules.proto.embedding.embedding_buffer_pb2 as embedding_pb2
import modules.proto.embedding.embedding_buffer_pb2_grpc as embedding_pb2_grpc

def get_json_stream(file_path):
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
            request = embedding_pb2.EmbeddingRequest(file_path=file_path)
            response_iterator = stub.StreamEmbedding(request)
            for response in response_iterator:
                print("Received response from server.")
                print("JSON stream:", response.json_stream)
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    test_file_path = "embed/story.txt"
    print("Streaming JSON response:")
    get_json_stream(test_file_path)