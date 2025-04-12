from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str
    temperature: float
    streaming: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass
class PathsConfig:
    session_tracking: Path
    checkpoint_db: Path
    summary_prompt: Path

    def to_dict(self):
        return asdict(self)

@dataclass
class TimeoutsConfig:
    idle_seconds: int
    watchdog_poll_seconds: int
    autorefresh_seconds: int

    def to_dict(self):
        return asdict(self)

@dataclass
class SummaryConfig:
    enabled: bool
    include_advice: bool
    include_next_steps: bool
    format_sections: bool

    def to_dict(self):
        return asdict(self)

@dataclass
class AppConfig:
    name: str
    port: int
    open_browser: bool
    environment: str

    def to_dict(self):
        return asdict(self)

@dataclass
class GraphConfig:
    retriever_type: str
    retriever_weights: list
    k: int
    device: str

    def to_dict(self):
        return asdict(self)
    
@dataclass
class FullConfig:
    app: AppConfig
    model: ModelConfig
    paths: PathsConfig
    timeouts: TimeoutsConfig
    summary: SummaryConfig
    graph: GraphConfig

    def to_dict(self):
        return asdict(self)

