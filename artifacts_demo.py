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


processed_data = Artifact(
  name="processed_data",
  # partition_keys=[key1, key2, ...]
)


@task(container_image=image)
def process_data_task() -> pd.DataFrame:
    entries = ["a", "b", "c", "d", "e"]
    numeric_data = [random.random() for _ in range(len(entries))]

    return pd.DataFrame({"col_a": entries, "col_b": numeric_data})


@workflow
def process_data_wf() -> Annotated[pd.DataFrame, processed_data]:
    return process_data_task()


trained_model = Artifact(
  name="trained_model",
  # partition_keys=[key1, key2, ...]
)

def generate_card(df: pd.DataFrame) -> str:
    contents = "# Dataset Card\n" "\n" "## Tabular Data\n"
    contents = contents + df.to_markdown()
    return contents


@task(container_image=image)
def train_model(data: pd.DataFrame, learning_rate: float, epochs: int) -> Annotated[FlyteFile, trained_model]:

    model = {
        "weights": [random.random() for _ in range(5)],
        "bias": random.random(),
        "training_time": random.randint(2, 5),
        "accuracy": round(random.uniform(0.7, 0.99), 2),
    }

    print(f"Model trained in {model['training_time']} seconds with accuracy {model['accuracy']}")

    file_name = "model.json"
    file_path = os.path.join(flytekit.current_context().working_directory, file_name)

    with open(file_path, 'w') as file:
        json.dump(model, file, indent=4)

    # return FlyteFile(path=file_path)

    return trained_model.create_from(
        FlyteFile(path=file_path),
        ModelCard(generate_card(pd.DataFrame(model)))
    )


data_query = processed_data.query()


@workflow
def train_model_wf(
        learning_rate: float,
        epochs: int,
        data: pd.DataFrame = data_query,
):
    train_model(data=data, learning_rate=learning_rate, epochs=epochs)


trigger_lp = LaunchPlan.get_or_create(
    name="trigger_train_model_lp",
    workflow=train_model_wf,
    trigger=OnArtifact(
        trigger_on=processed_data,
        inputs={
            "learning_rate": 0.01,
            "epochs": 10,
        }
    )
)