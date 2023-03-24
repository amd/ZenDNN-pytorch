import service_pb2_grpc
import grpc 
from concurrent import futures

class DynamoServicer(service_pb2_grpc.DynamoServicer):
    def Optimize(self, request, context):
        num = request.num
        print("Echo", num)
        return num

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_DynamoServicer_to_server(
        DynamoServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

serve()