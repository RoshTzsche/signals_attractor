from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    window_s: float = 60.0
    step_s: float = 10.0
    lead_s: float = 30.0
    horizon_s: float = 120.0
    min_sleep_s: float = 120.0
    min_wake_s: float = 60.0
    low_hz: float = 0.5
    high_hz: float = 35.0
    filter_order: int = 4
    warmup_s: float = 20.0
    tau_s: float = 0.05
    dimension: int = 4
    theiler_s: float = 1.0
    max_points: int = 256
    recurrence_radius: float = 0.5
    seed: int = 42

    def __post_init__(self):
        for name in ("dimension", "max_points", "filter_order", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in ("window_s", "step_s", "horizon_s", "min_sleep_s", "min_wake_s",
                     "low_hz", "high_hz", "tau_s", "recurrence_radius"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lead_s", "warmup_s", "theiler_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.high_hz <= self.low_hz or self.step_s > self.window_s:
            raise ValueError("Require low_hz < high_hz and step_s <= window_s")
        if self.dimension < 2 or self.max_points < 16 or self.filter_order < 1:
            raise ValueError("dimension >= 2, max_points >= 16 and filter_order >= 1 required")
        if self.dimension * self.tau_s + self.theiler_s >= self.window_s:
            raise ValueError("Embedding and Theiler window must fit in the analysis window")

    def to_dict(self):
        return asdict(self)
