"""Qwen3 text-only SFT and evaluation helpers."""

__all__ = ["LLMTuningConfig", "load_config"]


def __getattr__(name: str):
    if name in __all__:
        from .config import LLMTuningConfig, load_config

        return {"LLMTuningConfig": LLMTuningConfig, "load_config": load_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
