"""
xmad/2_iteration/xmad_debate/config.py

Typed loader for config/debate.yaml. Loading into dataclasses (rather than passing
a raw dict around) means a typo or missing field in the YAML fails immediately at
startup, with a clear error, instead of surfacing deep inside a multi-hour run.

    from xmad_debate.config import load_config
    cfg = load_config("config/debate.yaml")
    cfg.agents.agent_a.url
    cfg.decoding.temperature
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    label: str
    n_questions: int | None
    start: int


@dataclass
class DatasetConfig:
    name: str
    config: str
    split: str


@dataclass
class AgentEndpoint:
    alias: str
    url: str
    # Whether this model can emit more than one tool call in a single reply.
    # Llama-3.1's own official chat template hard-rejects >1 tool call per
    # turn ("This model only supports single tool-calls at once!"), so it
    # must be set to false for that model. Qwen has been observed batching
    # multiple searches into one turn (Phase 1); that behaviour is preserved
    # by default, since it is itself part of what the thesis measures.
    supports_parallel_tool_calls: bool = True
    # Optional per-agent override of debate.max_turns_per_round. Use this to
    # give a model that can only search sequentially (e.g. Llama, one search
    # per turn) more turns to reach the same amount of evidence a batching
    # model gathers in fewer turns. None = use the shared debate-wide default.
    max_turns_per_round: int | None = None


@dataclass
class AgentsConfig:
    agent_a: AgentEndpoint
    agent_b: AgentEndpoint


@dataclass
class DecodingConfig:
    temperature: float
    top_p: float
    # None = no cap passed to the API (the model generates until it stops on
    # its own). Used for the judge, where output length is naturally bounded
    # by the required Confidence/Final answer format.
    max_tokens: int | None = None


@dataclass
class DebateConfig:
    max_rounds: int
    consensus_check: str
    force_first_search: bool
    max_turns_per_round: int
    stalemate_patience: int = 1


@dataclass
class ToolsConfig:
    mcp_url: str
    agent_tools: list[str]


@dataclass
class JudgeConfig:
    enabled: bool
    url: str
    alias: str
    prompt_file: str
    decoding: DecodingConfig


@dataclass
class PromptsConfig:
    debate_system_prompt_file: str


@dataclass
class LoggingConfig:
    log_dir: str
    save_full_session: bool


@dataclass
class DebateExperimentConfig:
    run: RunConfig
    dataset: DatasetConfig
    agents: AgentsConfig
    decoding: DecodingConfig
    debate: DebateConfig
    tools: ToolsConfig
    judge: JudgeConfig
    prompts: PromptsConfig
    logging: LoggingConfig
    _base_dir: Path = field(default=Path("."), repr=False)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path from the config (e.g. a prompt file) relative to the
        config file's own directory, so the config works regardless of cwd."""
        return (self._base_dir / relative_path).resolve()

    def read_prompt(self, relative_path: str) -> str:
        return self.resolve(relative_path).read_text()


def load_config(path: str | Path) -> DebateExperimentConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = DebateExperimentConfig(
        run=RunConfig(**raw["run"]),
        dataset=DatasetConfig(**raw["dataset"]),
        agents=AgentsConfig(
            agent_a=AgentEndpoint(**raw["agents"]["agent_a"]),
            agent_b=AgentEndpoint(**raw["agents"]["agent_b"]),
        ),
        decoding=DecodingConfig(**raw["decoding"]),
        debate=DebateConfig(**raw["debate"]),
        tools=ToolsConfig(**raw["tools"]),
        judge=JudgeConfig(
            enabled=raw["judge"]["enabled"],
            url=raw["judge"]["url"],
            alias=raw["judge"]["alias"],
            prompt_file=raw["judge"]["prompt_file"],
            decoding=DecodingConfig(**raw["judge"]["decoding"]),
        ),
        prompts=PromptsConfig(**raw["prompts"]),
        logging=LoggingConfig(**raw["logging"]),
        # config/debate.yaml lives in config/; prompts/ and logs/ sit one level up,
        # at the project root, so relative paths in the YAML resolve from there.
        _base_dir=path.parent.parent,
    )
    return cfg