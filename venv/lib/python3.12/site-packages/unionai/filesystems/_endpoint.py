import os

import grpc

_ENDPOINT = "localhost:8080"
MAX_MESSAGE_LENGTH = 10 * 1024 * 1024


def _get_endpoint() -> str:
    endpoint = os.environ.get("OBJECT_STORE_ENDPOINT")
    if endpoint is not None and endpoint != "":
        return endpoint

    endpoint = os.environ.get("object_store_endpoint")
    if endpoint is not None and endpoint != "":
        return endpoint

    return _ENDPOINT


def _create_channel():
    return grpc.aio.insecure_channel(
        _get_endpoint(),
        options=[
            ("grpc.max_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
