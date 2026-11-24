from judgearena.datasets.registry import register_dataset
from judgearena.tasks.definitions.lmarena.dataset import LMArenaAdapter

register_dataset("lmarena")(LMArenaAdapter)
