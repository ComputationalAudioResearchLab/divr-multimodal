from typing import Literal
from .task_1 import database as task_1
from .task_2 import database as task_2

tasks = {"task_1": task_1, "task_2": task_2}
TASK_KEYS = Literal["task_1", "task_2"]
