import base64
import sys
import textwrap
from dataclasses import dataclass
from importlib.metadata import version
from os import environ
from typing import Optional

from flyteidl.service.identity_pb2 import UserInfoRequest
from flyteidl.service.identity_pb2_grpc import IdentityServiceStub
from flytekit.clients.auth_helper import get_authenticated_channel
from flytekit.configuration import AuthType, Config, PlatformConfig, get_config_file
from flytekit.configuration.default_images import DefaultImages

_GCP_SERVERLESS_ENDPOINT = "serverless-gcp.cloud-staging.union.ai"
_UNIONAI_SERVERLESS_API_KEY = "UNIONAI_SERVERLESS_API_KEY"
_SERVERLESS_IMAGE_REGISTRY: str = "us-central1-docker.pkg.dev/serverless-gcp-dataplane/union/unionai"


@dataclass
class _UnionAIConfig:
    serverless_endpoint: str = environ.get("UNIONAI_SERVERLESS_ENDPOINT", _GCP_SERVERLESS_ENDPOINT)
    org: Optional[str] = None
    config: Optional[str] = None


_UNIONAI_CONFIG = _UnionAIConfig()


@dataclass
class AppClientCredentials:
    endpoint: str
    client_id: str
    client_secret: str
    org: str


def _encode_app_client_credentials(app_credentials: AppClientCredentials) -> str:
    """Encode app_credentials with base64."""
    data = (
        f"{app_credentials.endpoint}:{app_credentials.client_id}:{app_credentials.client_secret}:{app_credentials.org}"
    )
    return base64.b64encode(data.encode("utf-8")).decode("utf-8")


def _decode_app_client_credentials(encoded_str: str) -> AppClientCredentials:
    """Decode encoded base64 string into app credentials."""
    endpoint, client_id, client_secret, org = base64.b64decode(encoded_str.encode("utf-8")).decode("utf-8").split(":")
    return AppClientCredentials(endpoint=endpoint, client_id=client_id, client_secret=client_secret, org=org)


def _get_config_obj(config: Optional[str] = None) -> Config:
    """Get Config object."""
    if config is None:
        config = _UNIONAI_CONFIG.config

    cfg_file = get_config_file(config)
    if cfg_file is None:
        serverless_api_value = environ.get(_UNIONAI_SERVERLESS_API_KEY, "")
        config = Config.for_endpoint(endpoint=_UNIONAI_CONFIG.serverless_endpoint)

        if serverless_api_value == "":
            # Serverless clients points to the serverless endpoint by default.
            return config
        try:
            app_credentials = _decode_app_client_credentials(serverless_api_value)
        except Exception as e:
            raise ValueError(f"Unable to read {_UNIONAI_SERVERLESS_API_KEY}") from e

        if _UNIONAI_CONFIG.org is None and app_credentials.org != "":
            _UNIONAI_CONFIG.org = app_credentials.org

        return config.with_params(
            platform=PlatformConfig(
                endpoint=app_credentials.endpoint,
                insecure=False,
                auth_mode=AuthType.CLIENTSECRET,
                client_id=app_credentials.client_id,
                client_credentials_secret=app_credentials.client_secret,
            )
        )
    else:
        # Allow for --config to still be passed in for Managed+ users.
        return Config.auto(config)


def _get_organization(platform_config: PlatformConfig) -> str:
    """Get organization based on endpoint."""
    if _UNIONAI_CONFIG.org is not None:
        return _UNIONAI_CONFIG.org
    elif platform_config.endpoint == _UNIONAI_CONFIG.serverless_endpoint:
        org = _get_user_handle(platform_config)
        _UNIONAI_CONFIG.org = org
        return org
    else:
        # Managed+ users, the org is not required for requests and we set it ""
        # to replicate default flytekit behavior.
        return ""


def _get_user_handle(platform_config: PlatformConfig) -> str:
    """Get user_handle for PlatformConfig."""
    with get_authenticated_channel(platform_config) as channel:
        client = IdentityServiceStub(channel)
        user_info = client.UserInfo(UserInfoRequest())
        user_handle = user_info.additional_claims.fields["userhandle"]
        return user_handle.string_value


def _get_default_image() -> str:
    """Get default image version."""
    cfg_obj = _get_config_obj(None)

    # TODO: This is only temporary to support GCP endpoints. When the unionai images are public,
    # we will always use unionai images
    if cfg_obj.platform.endpoint == _GCP_SERVERLESS_ENDPOINT:
        major, minor = sys.version_info.major, sys.version_info.minor
        unionai_version = version("unionai")
        if "dev" in unionai_version:
            suffix = "latest"
        else:
            suffix = unionai_version

        return f"{_SERVERLESS_IMAGE_REGISTRY}:py{major}.{minor}-{suffix}"

    return DefaultImages().find_image_for()


def _get_auth_success_html(endpoint: str) -> str:
    """Get default success html. Return None to use flytekit's default success html."""
    if endpoint.endswith("union.ai") or endpoint.endswith("unionai.cloud"):
        SUCCESS_HTML = textwrap.dedent(
            f"""
        <html>
        <head>
            <title>OAuth2 Authentication to UnionAI Successful</title>
        </head>
        <body style="background:white;font-family:Arial">
            <div style="position: absolute;top:40%;left:50%;transform: translate(-50%, -50%);text-align:center;">
                <div style="margin:auto">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 65" fill="currentColor"
                        style="color:#fdb51e;width:360px;">
                        <title>Union.ai</title>
                        <path d="M32,64.8C14.4,64.8,0,51.5,0,34V3.6h17.6v41.3c0,1.9,1.1,3,3,3h23c1.9,0,3-1.1,3-3V3.6H64V34
                        C64,51.5,49.6,64.8,32,64.8z M69.9,30.9v30.4h17.6V20c0-1.9,1.1-3,3-3h23c1.9,0,3,1.1,3,3v41.3H134V30.9c0-17.5-14.4-30.8-32.1-30.8
                        S69.9,13.5,69.9,30.9z M236,30.9v30.4h17.6V20c0-1.9,1.1-3,3-3h23c1.9,0,3,1.1,3,3v41.3H300V30.9c0-17.5-14.4-30.8-32-30.8
                        S236,13.5,236,30.9L236,30.9z M230.1,32.4c0,18.2-14.2,32.5-32.2,32.5s-32-14.3-32-32.5s14-32.1,32-32.1S230.1,14.3,230.1,32.4
                        L230.1,32.4z M213.5,20.2c0-1.9-1.1-3-3-3h-24.8c-1.9,0-3,1.1-3,3v24.5c0,1.9,1.1,3,3,3h24.8c1.9,0,3-1.1,3-3V20.2z M158.9,3.6
                        h-17.6v57.8h17.6V3.6z"></path>
                    </svg>
                    <h2>You've successfully authenticated to:
                        <br><em>{endpoint}</em>
                    </h2>
                    <p style="font-size:20px;">Return to your terminal for next steps</p>
                </div>
            </div>
        </body>
        </html>
        """  # noqa
        )
        return SUCCESS_HTML
    return None
