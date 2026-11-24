from judgearena.datasets.registry import register_dataset
from judgearena.tasks.definitions.comparia.dataset import ComparIAAdapter

register_dataset("comparia")(ComparIAAdapter)
