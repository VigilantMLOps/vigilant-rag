from pathlib import Path

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Obsidian vault
    vault_path: Path

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "obsidian_notes_v1_nomic_embed_text"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2:3b"

    # VigilantMLOps
    vigilant_api_url: str = "http://localhost:8000"
    emit_traces: bool = True

    @field_validator("vault_path")
    @classmethod
    def vault_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"VAULT_PATH does not exist: {v}")
        return v


def load_rag_config(path: Path | None = None) -> dict:
    config_path = path or Path(__file__).parent / "rag.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
