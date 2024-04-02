from collections import namedtuple
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import grpc
from flyteidl.service.admin_pb2_grpc import AdminServiceStub
from flyteidl.service.dataproxy_pb2_grpc import DataProxyServiceStub
from flyteidl.service.signal_pb2_grpc import SignalServiceStub
from flytekit.clients.auth_helper import get_authenticated_channel
from flytekit.configuration import PlatformConfig
from flytekit.remote import FlyteRemote


class ConcreteClientCallDetails(
    namedtuple(
        "ConcreteClientCallDetails",
        (
            "method",
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        ),
    ),
    grpc.ClientCallDetails,
):
    """Concrete implementation for ClientCallDetails."""


def set_org(obj: Any, org: str):
    """Recursively set the org in obj."""
    with suppress(AttributeError):
        obj.org = org

    with suppress(AttributeError):
        for _, value in obj.ListFields():
            set_org(value, org=org)

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        for value in obj:
            set_org(value, org=org)


class OrgInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Interceptor that adds org metadata into the grpc call."""

    def __init__(self, org: str):
        self.org = org

    def intercept_unary_unary(self, continuation, client_call_details, request):
        old_metadata = client_call_details.metadata or []
        new_metadata = old_metadata + [("x-user-active-org", self.org)]
        new_details = ConcreteClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=new_metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
            compression=client_call_details.compression,
        )

        # Inject org into every IDL requests
        set_org(request, org=self.org)
        return continuation(new_details, request)


def update_flyte_remote_with_interceptor(remote: FlyteRemote, org: str) -> FlyteRemote:
    """Updates FlyteRemote inplace with an interceptor with org."""
    if org == "":
        # There is no org, so no need to intercept
        return remote

    # Intercept calls to FlyteRemote and add org to grpc metadata
    org_interceptor = OrgInterceptor(org=org)
    flyte_client = remote.client
    previous_channel = flyte_client._channel
    new_channel = grpc.intercept_channel(previous_channel, org_interceptor)

    flyte_client._channel = new_channel
    flyte_client._stub = AdminServiceStub(new_channel)
    flyte_client._signal = SignalServiceStub(new_channel)
    flyte_client._dataproxy_stub = DataProxyServiceStub(new_channel)

    return remote


def get_authenticated_channel_with_org(platform_obj: PlatformConfig, org: str):
    channel = get_authenticated_channel(platform_obj)
    if org == "":
        return channel

    org_interceptor = OrgInterceptor(org=org)
    return grpc.intercept_channel(channel, org_interceptor)
