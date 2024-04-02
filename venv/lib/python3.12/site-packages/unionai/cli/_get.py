from typing import Optional

import rich_click as click
from grpc import RpcError
from rich.console import Console
from rich.table import Table

from unionai._config import _get_config_obj, _get_organization
from unionai._interceptor import get_authenticated_channel_with_org
from unionai.internal.common.list_pb2 import ListRequest
from unionai.internal.identity.app_payload_pb2 import ListAppsRequest
from unionai.internal.identity.app_service_pb2_grpc import AppsServiceStub
from unionai.internal.secret.payload_pb2 import ListSecretsRequest
from unionai.internal.secret.secret_pb2_grpc import SecretServiceStub


@click.group()
def get():
    """Get a resource."""


@get.command()
@click.pass_context
@click.option("--project", help="Project name")
@click.option("--domain", help="Domain name")
def secret(
    ctx: click.Context,
    project: Optional[str],
    domain: Optional[str],
):
    """Get secrets."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = SecretServiceStub(channel)
    secrets = []
    next_token, has_next = "", True

    try:
        while has_next:
            request = ListSecretsRequest(domain=domain, project=project, limit=20, token=next_token, organization=org)
            response = stub.ListSecrets(request)
            next_token = response.token
            has_next = next_token != ""

            secrets.extend(response.secrets)
    except RpcError as e:
        raise click.ClickException(f"Unable to get secrets.\n{e}") from e

    if secrets:
        table = Table()
        for name in ["name", "project", "domain"]:
            table.add_column(name, justify="right")

        for secret in secrets:
            project = secret.id.project or "-"
            domain = secret.id.domain or "-"
            table.add_row(secret.id.name, project, domain)

        console = Console()
        console.print(table)
    else:
        click.echo("No secrets found")


@get.command()
@click.pass_context
def app(ctx: click.Context):
    """Get applications."""
    platform_obj = _get_config_obj(ctx.obj.get("config_file", None)).platform
    org = _get_organization(platform_obj)

    channel = get_authenticated_channel_with_org(platform_obj, org)
    stub = AppsServiceStub(channel)

    apps = []
    next_token, has_next = "", True
    try:
        while has_next:
            request = ListAppsRequest(organization=org, request=ListRequest(limit=20, token=next_token))
            response = stub.List(request)
            next_token = response.token
            has_next = next_token != ""

            apps.extend(response.apps)
    except RpcError as e:
        raise click.ClickException(f"Unable to get apps.\n{e}") from e

    if apps:
        table = Table()
        for name in ["client_id", "client_name"]:
            table.add_column(name, overflow="fold")

        for app in apps:
            table.add_row(app.client_id, app.client_name)

        console = Console()
        console.print(table)
    else:
        click.echo("No apps found.")
