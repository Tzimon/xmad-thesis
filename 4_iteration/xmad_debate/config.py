"""
xmad/3_iteration/xmad_debate/config.py

Typed loader for one Phase 3 run config. phase3.py writes this file into each
run folder (results/<run>/run.yaml) from config/base.yaml + models.yaml +
runs.yaml, so a run folder always documents exactly what was run. A typo or
missing field fails at start-up, not deep inside a multi-hour run.

Differences to Phase 2:
  * AgentEndpoint carries the per-model reasoning switch (request_extra_body,
    system_prompt_suffix) and the model's registry key.
  * No judge here: in Phase 3 the judge runs as a separate pass after the
    debates (phase3.py judge), so a debate run never needs a judge server.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    run_id: str
    label: str
    n_questions: int
    seed: int


@dataclass
class DatasetConfig:
    name: str
    config: str
    split: str


@dataclass
class AgentEndpoint:
    model_key: str
    alias: str
    url: str
    gpus: list[int]
    supports_parallel_tool_calls: bool
    max_turns_per_round: int
    request_extra_body: dict = field(default_factory=dict)
    system_prompt_suffix: str = ""


@dataclass
class AgentsConfig:
    agent_a: AgentEndpoint
    agent_b: AgentEndpoint


@dataclass
class DecodingConfig:
    temperature: float
    top_p: float
    max_tokens: int | None = None


@dataclass
class DebateConfig:
    max_rounds: int
    force_first_search: bool
    stalemate_patience: int = 1


@dataclass
class ToolsConfig:
    mcp_url: str
    agent_tools: list[str]


@dataclass
class RunFileConfig:
    run: RunConfig
    dataset: DatasetConfig
    agents: AgentsConfig
    decoding: DecodingConfig
    debate: DebateConfig
    tools: ToolsConfig
    debate_system_prompt_file: str
    _base_dir: Path = field(default=Path("."), repr=False)

    def read_prompt(self, relative_path: str) -> str:
        return (self._base_dir / relative_path).read_text(encoding="utf-8")

    def system_prompt_for(self, agent: AgentEndpoint) -> str:
        base = self.read_prompt(self.debate_system_prompt_file)
        if not agent.system_prompt_suffix:
            return base
        return f"{base.rstrip()}\n\n{agent.system_prompt_suffix}"


def load_run_config(path: str | Path, base_dir: str | Path) -> RunFileConfig:
    """base_dir: the 3_iteration folder (prompt paths are relative to it)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunFileConfig(
        run=RunConfig(**raw["run"]),
        dataset=DatasetConfig(**raw["dataset"]),
        agents=AgentsConfig(
            agent_a=AgentEndpoint(**raw["agents"]["agent_a"]),
            agent_b=AgentEndpoint(**raw["agents"]["agent_b"]),
        ),
        decoding=DecodingConfig(**raw["decoding"]),
        debate=DebateConfig(**raw["debate"]),
        tools=ToolsConfig(**raw["tools"]),
        debate_system_prompt_file=raw["debate_system_prompt_file"],
        _base_dir=Path(base_dir),
    )
