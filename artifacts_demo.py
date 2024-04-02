import random
import time
import os
from typing_extensions import Annotated

import pandas as pd
from flytekit import task, workflow, LaunchPlan, ImageSpec
from flytekit.core.artifact import Artifact
from unionai.artifacts import OnArtifact


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
def process_data_task() -> Annotated[pd.DataFrame, processed_data]:
    entries = ["a", "b", "c", "d", "e"]
    numeric_data = [random.random() for _ in range(len(entries))]

    return pd.DataFrame({"col_a": entries, "col_b": numeric_data})


@workflow
def process_data_wf() -> pd.DataFrame:
    return process_data_task()


trained_model = Artifact(
  name="trained_model",
  # partition_keys=[key1, key2, ...]
)


@task(container_image=image)
def train_model(data: pd.DataFrame, learning_rate: float, epochs: int) -> Annotated[dict, trained_model]:

    start_time = time.time()
    time.sleep(random.randint(2, 5))

    model = {
        "weights": [random.random() for _ in range(5)],
        "bias": random.random(),
        "training_time": round(time.time() - start_time, 2),
        "accuracy": round(random.uniform(0.7, 0.99), 2),
    }

    print(f"Model trained in {model['training_time']} seconds with accuracy {model['accuracy']}")

    return model


data_query = processed_data.query()


@workflow
def train_model_wf(
        learning_rate: float,
        epochs: int,
        data: pd.DataFrame = data_query,
) -> dict:
    return train_model(data=data, learning_rate=learning_rate, epochs=epochs)


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