import json
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import click
from flytekit.core.constants import REQUIREMENTS_FILE_NAME
from flytekit.image_spec.image_spec import _F_IMG_ID, ImageBuildEngine, ImageSpec, ImageSpecBuilder
from flytekit.models.core import execution as core_execution_models


def _build(spec: Path, context: Optional[Path], target_image: str) -> str:
    """Build image using UnionAI."""
    from unionai._config import _GCP_SERVERLESS_ENDPOINT
    from unionai.configuration._plugin import UnionAIPlugin

    remote = UnionAIPlugin.get_remote(
        config=None,
        project="default",
        domain="development",
    )
    if remote.config.platform.endpoint != _GCP_SERVERLESS_ENDPOINT:
        raise ValueError(f"The UnionAI image builder only works with {_GCP_SERVERLESS_ENDPOINT}")

    context_url = "" if context is None else remote.upload_file(context)[1]
    spec_url = remote.upload_file(spec)[1]

    entity = remote.fetch_task(project="system", domain="production", name="build-image")

    start = datetime.now(timezone.utc)
    execution = remote.execute(
        entity,
        project="system",
        domain="development",
        inputs={"spec": spec_url, "context": context_url, "target_image": target_image},
    )

    console_url = remote.generate_console_url(execution)

    click.secho("👍 Build submitted!", bold=True, fg="yellow")
    click.secho(
        "⏳ Waiting for build to finish at: " + click.style(console_url, fg="cyan"),
        bold=True,
    )
    execution = remote.wait(execution, poll_interval=timedelta(seconds=1))

    elapsed = str(datetime.now(timezone.utc) - start).split(".")[0]
    if execution.closure.phase == core_execution_models.WorkflowExecutionPhase.SUCCEEDED:
        click.secho(f"✅ Build completed in {elapsed}!", bold=True, fg="green")
    else:
        click.secho(f"❌ Build failed in ! {elapsed}", bold=True, fg="red")
        raise RuntimeError("image build failed")
    return execution.outputs["fully_qualified_image"]


class UCImageSpecBuilder(ImageSpecBuilder):
    """ImageSpec builder for UnionAI."""

    _SUPPORTED_IMAGE_SPEC_PARAMETERS = {
        "name",
        "builder",
        "source_root",
        "env",
        "packages",
        "requirements",
        "apt_packages",
        "cuda",
        "cudnn",
        "platform",
        "pip_index",
        "commands",
    }

    def build_image(self, image_spec: ImageSpec):
        """Build image using UnionAI."""
        image_name = image_spec.image_name()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            spec_path, archive_path = self._validate_configuration(image_spec, tmp_path, image_name)

            fully_qualified_image = _build(spec_path, archive_path, image_name)
            if fully_qualified_image.startswith("cr.union.ai/"):
                # TODO: This replacement needs to happen in the mutating webhook.
                return fully_qualified_image.replace(
                    "cr.union.ai/",
                    "us-central1-docker.pkg.dev/serverless-gcp-dataplane/orgs/",
                )
            return fully_qualified_image

    def _validate_configuration(
        self, image_spec: ImageSpec, tmp_path: Path, image_name: str
    ) -> Tuple[Path, Optional[Path]]:
        """Validate and write configuration for builder."""
        unsupported_parameters = [
            name
            for name, value in vars(image_spec).items()
            if value is not None and name not in self._SUPPORTED_IMAGE_SPEC_PARAMETERS
        ]
        if unsupported_parameters:
            msg = f"The following parameters are unsupported and ignored: {unsupported_parameters}"
            warnings.warn(msg, UserWarning)

        # Hardcoded for now since our base image only supports 3.11
        spec = {"python_version": "3.11"}

        # Transform image spec into a spec we expect
        if image_spec.apt_packages:
            spec["apt_packages"] = image_spec.apt_packages
        if image_spec.commands:
            spec["commands"] = image_spec.commands
        if image_spec.cuda or image_spec.cudnn:
            spec["enable_gpu"] = True

        env = image_spec.env or {}
        env = {**{_F_IMG_ID: image_name}, **env}
        if env:
            spec["env"] = env
        packages = ["flytekit"]
        if image_spec.packages:
            packages.extend(image_spec.packages)
        spec["python_packages"] = packages
        if image_spec.pip_index:
            spec["pip_extra_index_urls"] = [image_spec.pip_index]

        context_path = tmp_path / "build.uc-image-builder"
        context_path.mkdir(exist_ok=True)

        if image_spec.requirements:
            shutil.copy2(image_spec.requirements, context_path / REQUIREMENTS_FILE_NAME)
            spec["python_requirements_files"] = [REQUIREMENTS_FILE_NAME]

        if image_spec.source_root:
            # Easter egg
            # Load in additional packages before installing pip/apt packages
            vendor_path = Path(image_spec.source_root) / ".vendor"
            if vendor_path.is_dir():
                spec["dist_dirpath"] = ".vendor"
            shutil.copytree(image_spec.source_root, context_path, dirs_exist_ok=True)

        if any(context_path.iterdir()):
            archive_path = Path(shutil.make_archive(tmp_path / "context", "xztar", context_path))
        else:
            archive_path = None

        spec_path = tmp_path / "spec.json"
        with spec_path.open("w") as f:
            json.dump(spec, f)

        return (spec_path, archive_path)


def _register_union_image_builder():
    ImageBuildEngine.register("ucimage", UCImageSpecBuilder(), priority=10)
