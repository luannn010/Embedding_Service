import os
import grpc
import modules.proto.embedding.embedding_buffer_pb2 as embedding_pb2
import modules.proto.embedding.embedding_buffer_pb2_grpc as embedding_pb2_grpc

def get_file_stream(file_path):
    """Read the file and return its content as bytes."""
    with open(file_path, "rb") as file:
        return file.read()

def save_json_stream(file_name, json_stream):
    """Save the JSON stream to the embed/embedded/ directory."""
    output_dir = "embed/embedded"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    # Extract the name from the file name (without extension)
    name, _ = os.path.splitext(os.path.basename(file_name))
    
    # Construct the output file path
    output_file_path = os.path.join(output_dir, f"{name}_embedded.json")
    
    # Save the JSON stream to the file
    with open(output_file_path, "w") as output_file:
        output_file.write(json_stream)
    
    print(f"JSON stream saved to {output_file_path}")

def get_json_stream(file_path, file_stream):
    """Send the file stream via gRPC and save the JSON response."""
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
            request = embedding_pb2.EmbeddingRequest(file_stream=file_stream)
            response_iterator = stub.StreamEmbedding(request)
            
            for response in response_iterator:
                print("Received response from server.")
                print("JSON stream:", response.json_stream)
                
                # Save the JSON stream to a file
                save_json_stream(file_path, response.json_stream)
                
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    test_file_path = "embed/stories/the_clock_maker.txt"
    file_stream = get_file_stream(test_file_path)
    print("Streaming JSON response:")
    get_json_stream(test_file_path, file_stream)
