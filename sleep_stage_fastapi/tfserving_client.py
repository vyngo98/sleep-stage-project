import grpc

from tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc
)

channel = grpc.insecure_channel(
    # "localhost:8500"
    "tfserving:8500" # when using docker compose
)

stub = (
    prediction_service_pb2_grpc
    .PredictionServiceStub(channel)
)