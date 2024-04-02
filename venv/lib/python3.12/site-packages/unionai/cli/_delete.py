from typing import Optional

import rich_click as click
from grpc import RpcError

from unionai._config import _get_config_obj, _get_organization
from unionai._interceptor import get_authenticated_channel_with_org
from unionai.internal.identity.app_payload_pb2 import DeleteAppRequest, GetAppRequest
from unionai.internal.identity.app_service_pb2_grpc import AppsServiceStub
from unionai.internal.secret.definition_pb2 import SecretIdentifier
from unionai.internal.secret.payload_pb2 import DeleteSecretRequest
from unionai.internal.secret.secret_pb2_grpc import SecretServiceStub


@click.group()
def delete():
    """Delete a resource."""


@delete.command()
@click.pass_context
@click.argument("name")
@click.option("--project", help="Project name")
@click.option("--domain", help="Domain name")
def secret(
    ctx: click.Context,
    name: str,
    project: Optional[str],
    domain: Optional[str],
):
    """Delete secret with NAME."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = SecretServiceStub(channel)

    request = DeleteSecretRequest(
        id=SecretIdentifier(name=name, domain=domain, project=project, organization=org),
    )
    try:
        stub.DeleteSecret(request)
        click.echo(f"Deleted secret with name: {name}")
    except RpcError as e:
        raise click.ClickException(f"Unable to delete secret with name: {name}\n{e}") from e


@delete.command()
@click.pass_context
@click.argument("client_id")
def app(
    ctx: click.Context,
    client_id: str,
):
    """Delete app with CLIENT_ID."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = AppsServiceStub(channel)

    # Get app first to make sure it exist
    get_request = GetAppRequest(organization=org, client_id=client_id)
    try:
        stub.Get(get_request)
    except RpcError as e:
        raise click.ClickException(
            f"Unable to delete app with client_id: {client_id} because it does not exist.\n{e}"
        ) from e

    delete_request = DeleteAppRequest(organization=org, client_id=client_id)
    try:
        stub.Delete(delete_request)
        click.echo(f"Deleted app with client_id: {client_id}")
    except RpcError as e:
        raise click.ClickException(f"Unable to delete app with client_id: {client_id}\n{e}") from e
