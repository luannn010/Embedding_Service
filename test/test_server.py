import unittest
import grpc
from concurrent import futures
from modules.proto.embedding import embedding_buffer_pb2, embedding_buffer_pb2_grpc
from modules.services import Labeler, EmbeddingGenerator
from main import EmbeddingService

class MockLabeler(Labeler):
    def create_definition(self, file_path):
        return '''
        {
            "collection_name": "test_collection",
            "partition_name": "test_partition",
            "description": "Test collection description",
            "dimension": 128,
            "metric_type": "L2"
        }
        '''

class MockEmbeddingGenerator(EmbeddingGenerator):
    def get_embedding(self, definition_json):
        return [0.1, 0.2, 0.3, 0.4, 0.5]

class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        self.service = EmbeddingService()
        self.service.labeler = MockLabeler()
        self.service.embedder = MockEmbeddingGenerator()
        embedding_buffer_pb2_grpc.add_EmbeddingServiceServicer_to_server(self.service, self.server)
        self.server.add_insecure_port('[::]:50051')
        self.server.start()
        self.fake_channel = grpc.insecure_channel('localhost:50051')

    def test_stream_embedding(self):
        stub = embedding_buffer_pb2_grpc.EmbeddingServiceStub(self.fake_channel)
        request = embedding_buffer_pb2.EmbeddingRequest(file_path="mock_file.txt")
        responses = stub.StreamEmbedding(request)
        response_list = list(responses)
        self.assertEqual(len(response_list), 1)
        response = response_list[0]
        
        # Use assertAlmostEqual for each element
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        for actual, expected in zip(response.embedding, expected_embedding):
            self.assertAlmostEqual(actual, expected, places=7)

        template = response.json_template
        self.assertIn("definition", template)
        self.assertIn("embeddings", template)

    def tearDown(self):
        self.server.stop(None)

if __name__ == '__main__':
    unittest.main()