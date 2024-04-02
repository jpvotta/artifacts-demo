from datetime import timedelta
from typing import Any, Dict, Optional, Type, Union

import isodate
from flyteidl.core import interface_pb2
from flytekit.core.artifact import Artifact, ArtifactQuery, Partition, TimePartition
from flytekit.core.context_manager import FlyteContextManager
from flytekit.core.launch_plan import LaunchPlan
from flytekit.core.schedule import LaunchPlanTriggerBase
from flytekit.core.type_engine import TypeEngine
from flytekit.models.interface import ParameterMap, Variable

from unionai.internal.artifacts import artifacts_pb2


class OnArtifact(LaunchPlanTriggerBase):
    """
     OnArtifact(
        trigger_on=dailyArtifact,
        inputs={
            "today_upstream": dailyArtifact,  # this means: use the matched artifact
            "yesterday_upstream": dailyArtifact.query(time_partition=dailyArtifact.time_partition - timedelta(days=1)),
            # this means: use the matched hourly artifact
            "other_daily_upstream": hourlyArtifact.query(partitions={"region": "LAX"}),
            "region": "SEA",  # static value that will be passed as input
            "other_artifact": UnrelatedArtifact.query(time_partition=dailyArtifact.time_partition - timedelta(days=1)),
            "other_artifact_2": UnrelatedArtifact.query(time_partition=hourlyArtifact.time_partition.truncate_to_day()),
            "other_artifact_3": UnrelatedArtifact.query(region=hourlyArtifact.time_partition.truncate_to_day()),
        },
    )
    """

    def __init__(self, trigger_on: Artifact, inputs: Optional[Dict[str, Union[Any, Artifact, ArtifactQuery]]] = None):
        super().__init__()
        if not isinstance(trigger_on, Artifact):
            raise ValueError("Trigger must be called with a single artifact")
        self.trigger_on = trigger_on
        self.inputs = inputs or {}  # user doesn't need to specify if the launch plan doesn't take inputs
        for k, v in self.inputs.items():
            if isinstance(v, ArtifactQuery):
                # This one binding reflects all the bindings across all the partition values as well.
                if v.binding:
                    if v.binding != self.trigger_on:
                        raise ValueError(
                            f"Binding {v.binding} id {id(v.binding)} must"
                            f"reference the triggering artifact {self.trigger_on}"
                        )

            if isinstance(v, Artifact):
                if v is not self.trigger_on:
                    raise ValueError(f"Binding {v} id {id(v)} must reference triggering artifact {self.trigger_on}")

    def get_parameter_map(
        self, input_python_interface: Dict[str, Type], input_typed_interface: Dict[str, Variable]
    ) -> interface_pb2.ParameterMap:
        """
        This is the key function that enables triggers to work. When declaring a trigger, the user specifies an input
        map in the form of artifacts, artifact time partitions, and artifact queries (possibly on unrelated artifacts).
        When it comes time to create the trigger, we need to convert all of these into a parameter map (because we've
        chosen Parameter as the method by which things like artifact queries are passed around). This function does
        that, and converts constants to Literals.
        """
        ctx = FlyteContextManager.current_context()
        pm = {}
        for k, v in self.inputs.items():
            var = input_typed_interface[k].to_flyte_idl()
            if isinstance(v, Artifact):
                aq = v.embed_as_query()
                p = interface_pb2.Parameter(var=var, artifact_query=aq)
            elif isinstance(v, ArtifactQuery):
                p = interface_pb2.Parameter(var=var, artifact_query=v.to_flyte_idl(binding=self.trigger_on))
            elif isinstance(v, TimePartition):
                expr = None
                if v.other and isinstance(v.other, timedelta):
                    expr = isodate.duration_isoformat(v.other)
                aq = v.reference_artifact.embed_as_query(bind_to_time_partition=True, expr=expr, op=v.op)
                p = interface_pb2.Parameter(var=var, artifact_query=aq)
            elif isinstance(v, Partition):
                # The reason is that if we bind to arbitrary partitions, we'll have to start keeping track of types
                # and if the experiment service is written in non-python, we'd have to reimplement a type engine in
                # the other language
                raise ValueError(
                    "Don't bind to non-time partitions. Time partitions are okay because of the known type."
                )
            else:
                lit = TypeEngine.to_literal(ctx, v, input_python_interface[k], input_typed_interface[k].type)
                p = interface_pb2.Parameter(var=var, default=lit.to_flyte_idl())

            pm[k] = p
        return interface_pb2.ParameterMap(parameters=pm)

    def to_flyte_idl(self, *args, **kwargs) -> artifacts_pb2.Trigger:
        lp: LaunchPlan = args[0]
        # project/domain will be empty - to be bound later at registration time.
        artifact_id = self.trigger_on.to_id_idl()

        pm = self.get_parameter_map(lp.python_interface.inputs, lp.interface.inputs)
        pm_model = ParameterMap.from_flyte_idl(pm)

        return artifacts_pb2.Trigger(
            trigger=artifact_id,
            trigger_inputs=pm_model.to_flyte_idl(),
        )
