from typing import Optional

import rich_click as click
from grpc import RpcError

from unionai._config import _get_config_obj, _get_organization
from unionai._interceptor import get_authenticated_channel_with_org
from unionai.internal.secret.definition_pb2 import SecretIdentifier, SecretSpec
from unionai.internal.secret.payload_pb2 import UpdateSecretRequest
from unionai.internal.secret.secret_pb2_grpc import SecretServiceStub


@click.group()
def update():
    """Update a resource."""


@update.command()
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
    """Update secret with NAME."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = SecretServiceStub(channel)
    request = UpdateSecretRequest(
        id=SecretIdentifier(name=name, domain=domain, project=project, organization=org),
        secret_spec=SecretSpec(value=value),
    )

    try:
        stub.UpdateSecret(request)
        click.echo(f"Updated secret with name: {name}")
    except RpcError as e:
        raise click.ClickException(f"Unable to update secret with name: {name}\n{e}") from e
