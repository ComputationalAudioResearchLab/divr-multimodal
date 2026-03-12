import yaml
import typing
from pathlib import Path
from typing import Literal
from divr_diagnosis import DiagnosisMap

from .GeneratorV1 import GeneratorV1

VERSIONS = Literal["v1"]
versions = typing.get_args(VERSIONS)
generator_map = {"v1": GeneratorV1()}


class YamlDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(YamlDumper, self).increase_indent(flow, False)


async def generate_tasks(
    version: VERSIONS,
    source_path: Path,
    diagnosis_map: DiagnosisMap,
    diag_level: int = 0,
    databases: list[str] | None = None,
    text_fields: list[str] | None = None,
    text_equals: list[str] | None = None,
    labels: list[str] | None = None,
    task_name: str | None = None,
) -> None:
    module_path = Path(__file__).parent.parent.resolve()
    output_name = task_name if task_name is not None else version
    tasks_path = Path(f"{module_path}/tasks/{output_name}")
    await generator_map[version](
        source_path=source_path,
        tasks_path=tasks_path,
        diagnosis_map=diagnosis_map,
        diag_level=diag_level,
        databases=databases,
        text_fields=text_fields,
        text_equals=text_equals,
        labels=labels,
    )


async def collect_diagnosis_terms(version: VERSIONS, source_path: Path):
    output_file_path = f"{source_path}/diag_terms.yml"
    terms = await generator_map[version].collect_diagnosis_terms(
        source_path=source_path
    )
    terms = {key: list(val) for key, val in terms.items()}
    with open(output_file_path, "w") as output_file:
        yaml.dump(terms, output_file, Dumper=YamlDumper, allow_unicode=True)
    print(f"terms collected into {output_file_path}")
