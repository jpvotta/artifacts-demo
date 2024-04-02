# TODO: Remove try/except when https://github.com/flyteorg/flytekit/pull/2136/ is merged
try:
    import flytekit.core.artifact  # noqa: F401 place this line first

    from unionai.artifacts._artifact import Artifact
    from unionai.artifacts._card import DataCard, ModelCard
    from unionai.artifacts._triggers import OnArtifact

    __all__ = ["Artifact", "OnArtifact", "DataCard", "ModelCard"]

except (ModuleNotFoundError, ImportError):
    pass
