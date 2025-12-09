"""CLI entry point for the Image Describer application."""

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import load_config
from .processor import process_images


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application.

    Args:
        debug: If True, set level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


app = typer.Typer(
    name="image-describer",
    help="Generate image descriptions with Ollama Vision.",
)


@app.command()
def main(
    image_folder: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to the folder containing images",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            help="Vision provider: 'ollama' or 'openai'",
        ),
    ] = None,
    suffix: Annotated[
        Optional[str],
        typer.Option(
            "--suffix",
            "-s",
            help="Text appended to each description (e.g., ', By Artist')",
        ),
    ] = None,
    prefix: Annotated[
        Optional[str],
        typer.Option(
            "--prefix",
            "-p",
            help="Text prepended to each description",
        ),
    ] = None,
    openai_base_url: Annotated[
        Optional[str],
        typer.Option(
            "--openai-url",
            help="OpenAI-compatible API base URL (e.g., https://api.openai.com/v1)",
        ),
    ] = None,
    openai_api_key: Annotated[
        Optional[str],
        typer.Option(
            "--openai-key",
            help="OpenAI-compatible API key",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            help="Overwrite existing .txt files (enabled by default)",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose mode",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            "-d",
            help="Enable debug logging (shows detailed API timing)",
        ),
    ] = False,
    web: Annotated[
        bool,
        typer.Option(
            "--web",
            "-w",
            help="Launch the web interface instead of CLI",
        ),
    ] = False,
) -> None:
    """Process images in a folder and generate descriptions."""
    # Setup logging early
    setup_logging(debug=debug)
    logger = logging.getLogger(__name__)

    if web:
        from .web_app import launch_web_app

        launch_web_app()
        return

    if image_folder is None:
        print("Error: Please provide an image folder path, or use --web for the web interface.")
        raise typer.Exit(1)

    if not image_folder.exists():
        print(f"Error: Folder not found: {image_folder}")
        raise typer.Exit(1)

    if not image_folder.is_dir():
        print(f"Error: Path is not a directory: {image_folder}")
        raise typer.Exit(1)

    cfg = load_config(
        config,
        suffix=suffix,
        prefix=prefix,
        provider=provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )
    
    # Log configuration
    if cfg.provider == "ollama":
        logger.info(f"Configuration loaded - Provider: Ollama, Model: {cfg.model}, Host: {cfg.ollama_host or 'default'}")
    else:
        logger.info(f"Configuration loaded - Provider: OpenAI-compatible, Model: {cfg.model}, URL: {cfg.openai_base_url or 'default'}")

    if verbose:
        print(f"Provider: {cfg.provider}")
        print(f"Model: {cfg.model}")
        if cfg.provider == "ollama":
            print(f"Ollama Host: {cfg.ollama_host}")
        else:
            print(f"API URL: {cfg.openai_base_url or 'default'}")
        print(f"Folder: {image_folder}")
        print(f"Extensions: {', '.join(cfg.supported_extensions)}")
        print("-" * 40)

    processed, skipped = process_images(
        folder=image_folder,
        config=cfg,
        overwrite=overwrite,
        verbose=verbose,
    )

    print("-" * 40)
    print(f"Done! {processed} image(s) processed, {skipped} skipped.")


if __name__ == "__main__":
    app()
