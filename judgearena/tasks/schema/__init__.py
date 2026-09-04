"""Public schema API for declarative task definitions.

The implementation is split by responsibility so dataset and evaluation
protocol schemas can evolve independently. Re-exporting the public models here
keeps existing imports stable.
"""

from judgearena.tasks.schema.baselines import (
    BaselineSpec,
    CategoryDefaultsBaseline,
    NoBaseline,
    OfficialOutputsBaseline,
    RuntimeRequiredBaseline,
    TaskDefaultBaseline,
)
from judgearena.tasks.schema.dataset import DatasetFields, DatasetSpec
from judgearena.tasks.schema.elo import EloProtocol, EloScoringSpec
from judgearena.tasks.schema.metrics import MetricSpec, ScoringSpec
from judgearena.tasks.schema.mt_bench import (
    MTBenchJudgeSpec,
    MTBenchProtocol,
    MultiTurnGeneration,
)
from judgearena.tasks.schema.pairwise import (
    PairwiseJudgeSpec,
    PairwiseProtocol,
    SingleTurnGeneration,
    SwapMode,
)
from judgearena.tasks.schema.resolved import (
    ResolvedTaskSpec,
    ResourceDigest,
    TaskProvenance,
    TaskSelection,
)
from judgearena.tasks.schema.sources import (
    GitRawSource,
    HuggingFaceDatasetSource,
    HuggingFaceSpaceSource,
    SourceSpec,
)
from judgearena.tasks.schema.task import (
    ProtocolSpec,
    SuffixVariants,
    TaskMetadata,
    TaskSpec,
)

__all__ = [
    "BaselineSpec",
    "CategoryDefaultsBaseline",
    "DatasetFields",
    "DatasetSpec",
    "EloProtocol",
    "EloScoringSpec",
    "GitRawSource",
    "HuggingFaceDatasetSource",
    "HuggingFaceSpaceSource",
    "MTBenchJudgeSpec",
    "MTBenchProtocol",
    "MetricSpec",
    "MultiTurnGeneration",
    "NoBaseline",
    "OfficialOutputsBaseline",
    "PairwiseJudgeSpec",
    "PairwiseProtocol",
    "ProtocolSpec",
    "ResolvedTaskSpec",
    "ResourceDigest",
    "RuntimeRequiredBaseline",
    "ScoringSpec",
    "SingleTurnGeneration",
    "SourceSpec",
    "SuffixVariants",
    "SwapMode",
    "TaskDefaultBaseline",
    "TaskMetadata",
    "TaskProvenance",
    "TaskSelection",
    "TaskSpec",
]
