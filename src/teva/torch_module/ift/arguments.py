from dataclasses import dataclass, field

from trl import ScriptArguments as BaseScriptArguments

from teva.teva_tasks import TevaTasks


@dataclass
class ScriptArguments(BaseScriptArguments):
    teva_tasks: list[TevaTasks]

    def __post_init__(self):
        self.teva_tasks = [TevaTasks[task.upper()] for task in self.teva_tasks]
