import random
import os
import json
from typing_extensions import Annotated

import pandas as pd
import flytekit
from flytekit import task, workflow, LaunchPlan, ImageSpec
from flytekit.types.file import FlyteFile
from flytekit.core.artifact import Artifact
from unionai.artifacts import OnArtifact
from unionai.artifacts import ModelCard


image = ImageSpec(
    builder="fast-builder",
    name="unionai-image",
    requirements="requirements.txt",
    registry=os.environ.get("DOCKER_REGISTRY", None),
)


blessed_model = Artifact(
  name="blessed_model",
  partition_keys=["blessed"]
)


def generate_card(df: pd.DataFrame) -> str:
    contents = "# Dataset Card\n" "\n" "## Tabular Data\n"
    contents = contents + df.to_markdown()
    return contents


@task(container_image=image)
def product_development_task(blessed_or_not: str) -> Annotated[dict, blessed_model]:
    model = {
        "weights": [random.random() for _ in range(5)],
        "bias": random.random(),
        "training_time": random.randint(2, 5),
        "accuracy": round(random.uniform(0.7, 0.99), 2),
    }

    return blessed_model.create_from(
        model,
        ModelCard(generate_card(pd.DataFrame(model))),
        blessed=blessed_or_not
    )


@workflow
def product_development_wf(blessed_or_not: str):
    product_development_task(blessed_or_not=blessed_or_not)


data_query = blessed_model.query(blessed="true")


prediction = Artifact(
  name="prediction",
)


@task(container_image=image)
def ops_task(model: dict) -> Annotated[FlyteFile, prediction]:

    file_name = "prediction.json"
    file_path = os.path.join(flytekit.current_context().working_directory, file_name)

    with open(file_path, 'w') as file:
        json.dump(model, file, indent=4)

    return prediction.create_from(
        FlyteFile(path=file_path)
    )


@workflow
def ops_wf(model: dict = data_query):
    ops_task(model=model)


trigger_lp = LaunchPlan.get_or_create(
    name="trigger_ops_wf_lp",
    workflow=ops_wf,
    trigger=OnArtifact(
        trigger_on=blessed_model,
    )
)