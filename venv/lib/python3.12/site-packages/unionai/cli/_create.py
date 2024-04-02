import re
from typing import Optional

import rich_click as click
from grpc import RpcError

from unionai._config import (
    _UNIONAI_CONFIG,
    AppClientCredentials,
    _encode_app_client_credentials,
    _get_config_obj,
    _get_organization,
    _get_user_handle,
)
from unionai._interceptor import get_authenticated_channel_with_org
from unionai.internal.identity.app_payload_pb2 import CreateAppRequest
from unionai.internal.identity.app_service_pb2_grpc import AppsServiceStub
from unionai.internal.identity.enums_pb2 import (
    ConsentMethod,
    GrantTypes,
    ResponseTypes,
    TokenEndpointAuthMethod,
)
from unionai.internal.secret.definition_pb2 import SecretIdentifier, SecretSpec
from unionai.internal.secret.payload_pb2 import CreateSecretRequest
from unionai.internal.secret.secret_pb2_grpc import SecretServiceStub


@click.group()
def create():
    """Create a resource."""


@create.command()
@click.pass_context
@click.argument("name")
@click.option("--value", prompt="Enter secret value", help="Secret value", hide_input=True)
@click.option("--project", help="Project name")
@click.option("--domain", help="Domain name")
def secret(
    ctx: click.Context,
    name: str,
    value: str,
    project: Optional[str],
    domain: Optional[str],
):
    """Create a secret with NAME."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)

    stub = SecretServiceStub(channel)
    request = CreateSecretRequest(
        id=SecretIdentifier(name=name, domain=domain, project=project, organization=org),
        secret_spec=SecretSpec(value=value),
    )
    try:
        stub.CreateSecret(request)
        click.echo(f"Created secret with name: {name}")
    except RpcError as e:
        raise click.ClickException(f"Unable to create secret with name: {name}\n{e}") from e


@create.command()
@click.pass_context
@click.argument("client_name")
def app(
    ctx: click.Context,
    client_name: str,
):
    """Create a app with CLIENT_NAME."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    normalized_client_name = re.sub("[^0-9a-zA-Z]+", "-", client_name.lower())
    if platform_obj.endpoint == _UNIONAI_CONFIG.serverless_endpoint:
        userhandle = _get_user_handle(platform_obj)
        tenant = _UNIONAI_CONFIG.serverless_endpoint.split(".")[0]
        client_id = f"{tenant}-{userhandle}-{normalized_client_name}"
    else:
        client_id = normalized_client_name
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = AppsServiceStub(channel)
    request = CreateAppRequest(
        organization=org,
        client_id=client_id,
        client_name=client_id,
        grant_types=[GrantTypes.CLIENT_CREDENTIALS, GrantTypes.AUTHORIZATION_CODE],
        redirect_uris=["http://localhost:8080/authorization-code/callback"],
        response_types=[ResponseTypes.CODE],
        token_endpoint_auth_method=TokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
        consent_method=ConsentMethod.CONSENT_METHOD_REQUIRED,
    )

    try:
        response = stub.Create(request)
    except RpcError as e:
        raise click.ClickException(f"Unable to create app with name: {client_name}\n{e}") from e

    click.echo(f"Client ID: {response.app.client_id}")
    click.echo("The following API key will only be shown once. Be sure to keep it safe!")
    click.echo("Configure your headless CLI by setting the following environment variable:")
    click.echo()

    serverless_api_key = _encode_app_client_credentials(
        AppClientCredentials(
            endpoint=platform_obj.endpoint,
            client_id=response.app.client_id,
            client_secret=response.app.client_secret,
            org=org,
        )
    )
    click.echo(f'export UNIONAI_SERVERLESS_API_KEY="{serverless_api_key}"')
