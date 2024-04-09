# force update v2

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


def generate_card(df: pd.DataFrame) -> str:
    contents = "# Sample Model Card\n" "\n" "## Tabular Data\n"
    contents = contents + df.to_markdown()
    return contents


blessed_model = Artifact(
  name="blessed_model",
  partition_keys=["blessed"]
)


@task(container_image=image)
def product_development_task(blessed_or_not: str) -> Annotated[FlyteFile, blessed_model]:
    model = {
        "weights": [random.random() for _ in range(5)],
        "bias": random.random(),
        "training_time": random.randint(2, 5),
        "accuracy": round(random.uniform(0.7, 0.99), 2),
    }

    file_name = "model.json"
    file_path = os.path.join(flytekit.current_context().working_directory, file_name)

    with open(file_path, 'w') as file:
        json.dump(model, file, indent=4)

    return blessed_model.create_from(
        FlyteFile(path=file_path),
        ModelCard(generate_card(pd.DataFrame(model))),
        blessed=blessed_or_not
    )


@workflow
def product_development_wf(blessed_or_not: str):
    product_development_task(blessed_or_not=blessed_or_not)


prediction = Artifact(
  name="prediction",
)


data_query = blessed_model.query(blessed="true")


@task(container_image=image)
def ops_task(model: FlyteFile) -> Annotated[FlyteFile, prediction]:

    with open(model, 'r') as file:
        content = file.read()

    content_dict = json.loads(content)
    processed_content = json.dumps(content_dict, indent=4)

    file_name = "prediction.json"
    file_path = os.path.join(flytekit.current_context().working_directory, file_name)
    with open(file_path, 'w') as file:
        file.write(processed_content)

    return prediction.create_from(
        FlyteFile(path=file_path)
    )


@workflow
def ops_wf(model: FlyteFile = data_query):
    ops_task(model=model)


trigger_lp = LaunchPlan.get_or_create(
    name="trigger_ops_wf_lp",
    workflow=ops_wf,
    trigger=OnArtifact(
        trigger_on=blessed_model,
    )
)