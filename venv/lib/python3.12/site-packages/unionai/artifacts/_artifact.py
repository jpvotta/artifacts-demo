from __future__ import annotations

import logging
import random
import typing
from datetime import timedelta
from typing import Optional, Union
from uuid import UUID

import isodate
from flyteidl.core import artifact_id_pb2 as art_id
from flyteidl.core.artifact_id_pb2 import Granularity
from flytekit.core.artifact import Artifact as FlytekitArtifact
from flytekit.core.artifact import (
    ArtifactQuery,
    ArtifactSerializationHandler,
    Partition,
    Partitions,
    Serializer,
    TimePartition,
)
from flytekit.models.literals import Literal
from flytekit.models.types import LiteralType

from unionai.internal.artifacts import artifacts_pb2

if typing.TYPE_CHECKING:
    from unionai.remote import UnionRemote

logger = logging.getLogger("unionai.artifacts")


class Artifact(FlytekitArtifact):
    """
    This is a wrapper around the Flytekit Artifact class.

    This Python class has two purposes - as a Python representation of a materialized Artifact,
    and as a way for users to specify that tasks/workflows create Artifacts and the manner
    in which they are created.

    Use one as input to workflow (only workflow for now)
    df_artifact = Artifact.get("flyte://a1")
    remote.execute(wf, inputs={"a": df_artifact})

    Note that Python fields will be missing when retrieved from the service.
    """

    def __init__(
        self,
        *args,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        time_partitioned: bool = False,
        time_partition: Optional[TimePartition] = None,
        time_partition_granularity: Optional[Granularity] = None,
        partition_keys: Optional[typing.List[str]] = None,
        partitions: Optional[Union[Partitions, typing.Dict[str, str]]] = None,
        python_val: Optional[typing.Any] = None,
        python_type: Optional[typing.Type] = None,
        literal: Optional[Literal] = None,
        literal_type: Optional[LiteralType] = None,
        short_description: Optional[str] = None,
        source: Optional[artifacts_pb2.ArtifactSource] = None,
        **kwargs,
    ):
        """
        :param project: Should not be directly user provided, the project/domain will come from the project/domain of
           the execution that produced the output. These values will be filled in automatically when retrieving however.
        :param domain: See above.
        :param name: The name of the Artifact. This should be user provided.
        :param version: Version of the Artifact, typically the execution ID, plus some additional entropy.
           Not user provided.
        :param time_partitioned: Whether or not this Artifact will have a time partition.
        :param time_partition: If you want to manually pass in the full TimePartition object
        :param time_partition_granularity: If you don't want to manually pass in the full TimePartition object, but
            want to control the granularity when one is automatically created for you. Note that consistency checking
            is limited while in alpha.
        :param partition_keys: This is a list of keys that will be used to partition the Artifact. These are not the
           values. Values are set via a () on the artifact and will end up in the partition_values field.
        :param partitions: This is a dictionary of partition keys to values.
        :param python_val: The Python value.
        :param python_type: The Python type.
        """
        self.python_val = python_val
        self.python_type = python_type
        self.literal = literal
        self.literal_type = literal_type
        self.short_description = short_description
        self.source = source
        super().__init__(
            project=project,
            domain=domain,
            name=name,
            version=version,
            time_partitioned=time_partitioned,
            time_partition=time_partition,
            time_partition_granularity=time_partition_granularity,
            partition_keys=partition_keys,
            partitions=partitions,
        )

    @classmethod
    def initialize(
        cls,
        python_val: typing.Any,
        python_type: typing.Type,
        name: Optional[str] = None,
        literal_type: Optional[LiteralType] = None,
        tags: Optional[typing.List[str]] = None,
    ) -> Artifact:
        """
        Use this for when you have a Python value you want to get an Artifact object out of.

        This function readies an Artifact for creation, it doesn't actually create it just yet since this is a
        network-less call. You will need to persist it with a FlyteRemote instance:
            remote.create_artifact(Artifact.initialize(...))

        Artifact.initialize("/path/to/file", tags={"tag1": "val1"})
        Artifact.initialize("/path/to/parquet", type=pd.DataFrame, tags=["0.1.0"])

        What's set here is everything that isn't set by the server. What is set by the server?
        - name, version, if not set by user.
        - uri
        Set by remote
        - project, domain
        """
        # Create the artifact object
        return Artifact(
            python_val=python_val,
            python_type=python_type,
            literal_type=literal_type,
            tags=tags,
            name=name,
        )

    @classmethod
    def from_flyte_idl(cls, pb2: artifacts_pb2.Artifact) -> Artifact:
        """
        Converts the IDL representation to this object.
        """
        a = Artifact(
            project=pb2.artifact_id.artifact_key.project,
            domain=pb2.artifact_id.artifact_key.domain,
            name=pb2.artifact_id.artifact_key.name,
            version=pb2.artifact_id.version,
            literal_type=LiteralType.from_flyte_idl(pb2.spec.type),
            literal=Literal.from_flyte_idl(pb2.spec.value),
            source=pb2.source,
        )
        if pb2.artifact_id.HasField("partitions"):
            if len(pb2.artifact_id.partitions.value) > 0:
                # static values should be the only ones set since currently we don't from_flyte_idl
                # anything that's not a materialized artifact.
                # if TIME_PARTITION in pb2.artifact_id.partitions.value:
                #     a._time_partition = TimePartition(pb2.artifact_id.partitions.value[TIME_PARTITION].static_value)
                #     a._time_partition.reference_artifact = a

                a._partitions = Partitions(
                    partitions={k: Partition(value=v, name=k) for k, v in pb2.artifact_id.partitions.value.items()}
                )
                a.partitions.reference_artifact = a
        if pb2.artifact_id.HasField("time_partition"):
            ts = pb2.artifact_id.time_partition.value.time_value
            dt = ts.ToDatetime()
            a._time_partition = TimePartition(
                dt,
                granularity=pb2.artifact_id.time_partition.granularity,
            )
            a._time_partition.reference_artifact = a

        return a

    def __str__(self):
        tp_str = f"  time partition={self.time_partition}\n" if self.time_partitioned else ""
        return (
            f"Artifact: project={self.project}, domain={self.domain}, name={self.name}, version={self.version}\n"
            f"  name={self.name}\n"
            f"  partitions={self.partitions}\n"
            f"{tp_str}"
            f"  literal_type="
            f"{self.literal_type}, "
            f"literal={self.literal})"
        )

    @classmethod
    def get(
        cls,
        uri: Optional[str],
        artifact_id: Optional[art_id.ArtifactID],
        remote: "UnionRemote",
        get_details: bool = False,
    ) -> Optional[Artifact]:
        """
        Use one locally. This retrieves the Literal.
        a = remote.get("flyte://blah")
        a = Artifact.get("flyte://blah", remote, tag="latest")
        u = union.get("union://blah")
        """
        return remote.get_artifact(uri=uri, artifact_id=artifact_id, get_details=get_details)

    @staticmethod
    def to_create_request(a: Artifact) -> artifacts_pb2.CreateArtifactRequest:
        if not a.project or not a.domain:
            raise ValueError("Project and domain are required to create an artifact")
        name = a.name or UUID(int=random.getrandbits(128)).hex
        ak = art_id.ArtifactKey(project=a.project, domain=a.domain, name=name)

        spec = artifacts_pb2.ArtifactSpec(
            value=a.literal,
            type=a.literal_type,
        )
        partitions = Serializer.partitions_to_idl(a.partitions)

        tp = None
        if a._time_partition:
            tv = a.time_partition.value.time_value
            if not tv:
                raise Exception("missing time value")
            tp = a.time_partition.value.time_value

        return artifacts_pb2.CreateArtifactRequest(
            artifact_key=ak, spec=spec, partitions=partitions, time_partition_value=tp
        )


class UnionArtifactSerializationHandler(ArtifactSerializationHandler):
    def partitions_to_idl(self, p: Optional[Partitions], **kwargs) -> Optional[art_id.Partitions]:
        if not p or not p.partitions:
            return None

        binding: Optional[Artifact] = kwargs.get("binding", None)
        if binding:
            return self.get_partitions_idl_with_binding(p, binding)

        pp = {}
        for k, v in p.partitions.items():
            if v.value is None:
                # This should only happen when serializing for triggers
                # Probably indicative of something in the data model that can be fixed
                # down the road.
                pp[k] = art_id.LabelValue(static_value="")
            else:
                pp[k] = v.value
        return art_id.Partitions(value=pp)

    @staticmethod
    def get_partitions_idl_with_binding(p: Partitions, binding: Artifact) -> art_id.Partitions:
        pp = {}
        # First create partition requirements for all the partitions
        if p.reference_artifact and p.reference_artifact == binding:
            triggering_artifact = binding
            if triggering_artifact.partition_keys:
                for k in triggering_artifact.partition_keys:
                    pp[k] = art_id.LabelValue(
                        triggered_binding=art_id.ArtifactBindingData(
                            partition_key=k,
                        )
                    )

        for k, v in p.partitions.items():
            if not v.reference_artifact or (
                v.reference_artifact
                and v.reference_artifact is p.reference_artifact
                and v.reference_artifact != binding
            ):
                # consider changing condition to just check for static value
                pp[k] = art_id.LabelValue(static_value=v.value.static_value)
            elif v.reference_artifact == binding:
                # This line here is why the PartitionValue object has a name field.
                # We might bind to a partition key that's a different name than the k here.
                pp[k] = art_id.LabelValue(
                    triggered_binding=art_id.ArtifactBindingData(
                        partition_key=v.name,
                    )
                )
            else:
                raise ValueError(f"Partition has unhandled reference artifact {v.reference_artifact}")

        return art_id.Partitions(value=pp)

    def time_partition_to_idl(self, tp: TimePartition, **kwargs) -> Optional[art_id.TimePartition]:
        binding: Optional[Artifact] = kwargs.get("binding", None)

        if binding:
            return self.get_time_partition_idl_with_binding(tp, binding)

        if not tp.value:
            # This is only for triggers - the backend needs to know of the existence of a time partition
            return art_id.TimePartition()

        return art_id.TimePartition(value=tp.value, granularity=tp.granularity)

    @staticmethod
    def get_time_partition_idl_with_binding(tp: TimePartition, binding: Artifact) -> art_id.TimePartition:
        if not tp.reference_artifact or (tp.reference_artifact and tp.reference_artifact != binding):
            # basically if there's no reference artifact, or if the reference artifact isn't
            # in the list of triggers, then treat it like normal.
            return art_id.TimePartition(value=tp.value, granularity=tp.granularity)
        elif tp.reference_artifact == binding:
            time_transform = None
            if tp.other and isinstance(tp.other, timedelta):
                expr = isodate.duration_isoformat(tp.other)
                time_transform = art_id.TimeTransform(transform=expr, op=tp.op)
            lv = art_id.LabelValue(
                triggered_binding=art_id.ArtifactBindingData(
                    bind_to_time_partition=True,
                    time_transform=time_transform,
                )
            )
            return art_id.TimePartition(value=lv, granularity=tp.granularity)
        # investigate if this happens, if not, remove. else
        logger.warning(f"Investigate - time partition in trigger with unhandled reference artifact {tp}")
        raise ValueError("Time partition reference artifact not found in ")

    def artifact_query_to_idl(self, aq: ArtifactQuery, **kwargs) -> art_id.ArtifactQuery:
        binding: Optional[Artifact] = kwargs.get("binding", None)

        ak = art_id.ArtifactKey(
            name=aq.name,
            project=aq.project,
            domain=aq.domain,
        )

        p = Serializer.partitions_to_idl(aq.partitions, binding=binding)
        tp = None
        if aq.time_partition:
            tp = aq.time_partition.to_flyte_idl(binding=binding)

        i = art_id.ArtifactID(
            artifact_key=ak,
            partitions=p,
            time_partition=tp,
        )

        aq = art_id.ArtifactQuery(
            artifact_id=i,
        )

        return aq


logger.debug("Overriding artifact serializer with Union's")
Serializer.register_serializer(UnionArtifactSerializationHandler())
