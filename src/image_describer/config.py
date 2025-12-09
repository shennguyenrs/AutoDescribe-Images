"""Configuration loading module."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

# User preferences file location (in config directory)
USER_PREFS_FILE = Path(__file__).parent.parent.parent / "config" / "user_preferences.json"


@dataclass
class UserPreferences:
    """User preferences that persist between sessions."""

    image_folder: str = ""
    provider: str = "ollama"
    ollama_model: str = ""
    openai_model: str = ""
    preset_key: str = ""
    prefix: str = ""
    suffix: str = ""
    temperature: float = 0.7
    overwrite: bool = True


def load_user_preferences() -> UserPreferences:
    """Load user preferences from JSON file.

    Returns:
        UserPreferences object with saved values or defaults.
    """
    if USER_PREFS_FILE.exists():
        try:
            with open(USER_PREFS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return UserPreferences(
                image_folder=data.get("image_folder", ""),
                provider=data.get("provider", "ollama"),
                ollama_model=data.get("ollama_model", ""),
                openai_model=data.get("openai_model", ""),
                preset_key=data.get("preset_key", ""),
                prefix=data.get("prefix", ""),
                suffix=data.get("suffix", ""),
                temperature=data.get("temperature", 0.7),
                overwrite=data.get("overwrite", True),
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return UserPreferences()


def save_user_preferences(prefs: UserPreferences) -> None:
    """Save user preferences to JSON file.

    Args:
        prefs: UserPreferences object to save.
    """
    data = {
        "image_folder": prefs.image_folder,
        "provider": prefs.provider,
        "ollama_model": prefs.ollama_model,
        "openai_model": prefs.openai_model,
        "preset_key": prefs.preset_key,
        "prefix": prefs.prefix,
        "suffix": prefs.suffix,
        "temperature": prefs.temperature,
        "overwrite": prefs.overwrite,
    }
    with open(USER_PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@dataclass
class PromptPreset:
    """A prompt preset configuration."""

    key: str
    name: str
    prompt: str
    markdown_format: bool = False


@dataclass
class Config:
    """Application configuration."""

    # Provider settings
    provider: str = "ollama"
    
    # Common settings
    model: str = "qwen3-vl:4b"
    system_prompt: str = "Describe this image in detail."
    temperature: float = 0.7
    supported_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    )
    description_suffix: str = ""
    description_prefix: str = ""
    markdown_format: bool = False
    
    # Ollama-specific settings
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = 8192  # Context window size (important for vision models with large images)
    
    # OpenAI-compatible settings
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    max_tokens: int = 8192


def load_presets(config_path: Path | None = None) -> list[PromptPreset]:
    """Load prompt presets from YAML configuration.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        List of PromptPreset objects.
    """
    if config_path is None:
        config_path = Path("config/config.yaml")

    presets = []

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        prompts_data = data.get("prompts", {})
        for key, preset_data in prompts_data.items():
            presets.append(
                PromptPreset(
                    key=key,
                    name=preset_data.get("name", key),
                    prompt=preset_data.get("prompt", "Describe this image."),
                    markdown_format=preset_data.get("markdownFormat", False),
                )
            )

    if not presets:
        presets.append(
            PromptPreset(
                key="default",
                name="Default",
                prompt="Describe this image in detail.",
            )
        )

    return presets


def get_ollama_models(ollama_host: str | None = None) -> list[str]:
    """Get available Ollama models.

    Args:
        ollama_host: Ollama server host URL.

    Returns:
        List of model names.
    """
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        client = Client(host=host)
        response = client.list()
        models = [model.model for model in response.models]
        return sorted(models)
    except Exception:
        return ["qwen3-vl:4b", "llava", "llama3.2-vision"]


def test_ollama_connection(ollama_host: str | None = None) -> tuple[bool, str]:
    """Test connection to Ollama server.

    Args:
        ollama_host: Ollama server host URL.

    Returns:
        Tuple of (success, message).
    """
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        client = Client(host=host)
        response = client.list()
        model_count = len(response.models)
        return True, f"✓ Connected to Ollama! Found {model_count} models."
    except Exception as e:
        return False, f"✗ Connection failed: {str(e)}"


def load_config(
    config_path: Path | None = None,
    suffix: str | None = None,
    prefix: str | None = None,
    preset_key: str | None = None,
    provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> Config:
    """Load configuration from .env and optional YAML file.

    Priority (highest to lowest):
        1. CLI arguments (suffix, prefix, preset_key, provider, etc.)
        2. Environment variables (.env)
        3. YAML config file
        4. Default values

    Args:
        config_path: Path to YAML configuration file.
        suffix: Suffix to append to descriptions (from CLI).
        prefix: Prefix to prepend to descriptions (from CLI).
        preset_key: Key of the preset to use.
        provider: Provider to use ('ollama' or 'openai').
        openai_base_url: OpenAI-compatible API base URL.
        openai_api_key: OpenAI-compatible API key.

    Returns:
        Config: The loaded configuration object.
    """
    # Start with defaults
    cfg_provider = provider or Config.provider
    ollama_host = Config.ollama_host
    model = Config.model
    system_prompt = Config.system_prompt
    temperature = Config.temperature
    num_ctx = Config.num_ctx
    max_tokens = Config.max_tokens
    supported_extensions = Config().supported_extensions
    description_suffix = Config.description_suffix
    description_prefix = Config.description_prefix
    markdown_format = Config.markdown_format

    # Load from YAML if exists
    if config_path is None:
        config_path = Path("config/config.yaml")

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Load defaults from YAML
        defaults = data.get("defaults", {})
        if provider is None:  # Only use YAML provider if not specified in CLI
            cfg_provider = defaults.get("provider", cfg_provider)
        temperature = defaults.get("temperature", temperature)
        model = defaults.get("model", model)
        num_ctx = defaults.get("num_ctx", num_ctx)
        max_tokens = defaults.get("max_tokens", max_tokens)
        supported_extensions = data.get("supported_extensions", supported_extensions)

        # Load prompt from preset if specified
        if preset_key:
            prompts_data = data.get("prompts", {})
            if preset_key in prompts_data:
                system_prompt = prompts_data[preset_key].get("prompt", system_prompt)
                markdown_format = prompts_data[preset_key].get("markdownFormat", False)
        else:
            # Fallback to old system_prompt format or first preset
            if "system_prompt" in data:
                system_prompt = data["system_prompt"]
            else:
                prompts_data = data.get("prompts", {})
                if prompts_data:
                    first_key = next(iter(prompts_data))
                    system_prompt = prompts_data[first_key].get("prompt", system_prompt)
                    markdown_format = prompts_data[first_key].get("markdownFormat", False)

    # Override with environment variables
    ollama_host = os.getenv("OLLAMA_HOST", ollama_host)
    if cfg_provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", model)
    
    # OpenAI settings from environment or arguments
    cfg_openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
    cfg_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if cfg_provider == "openai" and not openai_api_key:
        # Try to get model from env for OpenAI
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            model = env_model

    # Override with CLI arguments if provided
    if suffix is not None:
        description_suffix = suffix
    if prefix is not None:
        description_prefix = prefix

    return Config(
        provider=cfg_provider,
        ollama_host=ollama_host,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        supported_extensions=supported_extensions,
        description_suffix=description_suffix,
        description_prefix=description_prefix,
        markdown_format=markdown_format,
        openai_base_url=cfg_openai_base_url,
        openai_api_key=cfg_openai_api_key,
    )
