import service_pb2_grpc
import service_pb2
import grpc 

channel = grpc.insecure_channel('localhost:50051')
stub = service_pb2_grpc.DynamoStub(channel)
inp = service_pb2.TestInput(num=3)

result = stub.Optimize(inp)
print(result.num)