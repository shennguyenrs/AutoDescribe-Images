"""Web interface using Gradio."""

import base64
import os
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from .config import (
    Config,
    UserPreferences,
    get_ollama_models,
    test_ollama_connection,
    load_presets,
    load_user_preferences,
    save_user_preferences,
)
from .vision_client import list_models, test_connection
from .processor import find_images, process_images_generator


# Global stop event for cancellation
_stop_event = threading.Event()


def get_ollama_host() -> str:
    """Get the Ollama API host from environment or default."""
    host = os.getenv("OLLAMA_HOST", "")
    if not host:
        return "http://localhost:11434"
    # Replace 0.0.0.0 with localhost for better compatibility
    host = host.replace("0.0.0.0", "localhost")
    # Add http:// if missing
    if not host.startswith("http"):
        host = f"http://{host}"
    # Add default port if missing
    if host.count(":") == 1:  # Only has http:// but no port
        host = f"{host}:11434"
    return host


def create_app() -> gr.Blocks:
    """Create the Gradio web application."""

    presets = load_presets()
    preset_choices = [(p.name, p.key) for p in presets]
    preset_prompts = {p.key: p.prompt for p in presets}
    preset_markdown_format = {p.key: p.markdown_format for p in presets}

    # Load saved user preferences
    user_prefs = load_user_preferences()

    # Determine default preset
    default_preset = user_prefs.preset_key if user_prefs.preset_key else (preset_choices[0][1] if preset_choices else None)

    # Get logo as base64
    logo_path = Path(__file__).parent.parent.parent / "assets" / "logo.png"
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    with gr.Blocks(title="AutoDescribe Images") as app:
        gr.HTML(
            f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{logo_base64}"></div>'
        )
        gr.Markdown("Generate image descriptions using AI vision models.")
        
        # Provider selection
        provider_radio = gr.Radio(
            choices=[("Ollama (Local)", "ollama"), ("OpenAI-Compatible API", "openai")],
            value=user_prefs.provider,
            label="Vision Provider",
            info="Choose between local Ollama or OpenAI-compatible API (OpenAI, OpenRouter, etc.)",
        )
        
        connection_status = gr.HTML(
            value="<span style='color: gray;'>[Not Connected]</span>",
        )

        # Ollama settings (shown when provider is ollama)
        with gr.Group(visible=(user_prefs.provider == "ollama")) as ollama_group:
            ollama_host_input = gr.Textbox(
                label="Ollama API URL",
                value=get_ollama_host(),
                placeholder="http://localhost:11434",
                info="Ollama server URL",
            )
            connect_btn = gr.Button("🔄 Test Connection", size="sm")
        
        # OpenAI settings (shown when provider is openai)
        with gr.Group(visible=(user_prefs.provider == "openai")) as openai_group:
            openai_base_url_input = gr.Textbox(
                label="API Base URL",
                value="",
                placeholder="https://api.openai.com/v1 or https://openrouter.ai/api/v1",
                info="OpenAI-compatible API endpoint",
            )
            openai_api_key_input = gr.Textbox(
                label="API Key",
                value="",
                placeholder="sk-...",
                type="password",
                info="⚠️ Not saved for security. Re-enter each session.",
            )
            connect_openai_btn = gr.Button("🔄 Test Connection & Load Models", size="sm")

        folder_input = gr.Textbox(
            label="Image Folder",
            value=user_prefs.image_folder,
            placeholder="Enter path to folder containing images...",
            info="Path to the folder with images to process",
            lines=2,
        )

        with gr.Group():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[],
                value=user_prefs.ollama_model if user_prefs.provider == "ollama" else user_prefs.openai_model,
                allow_custom_value=True,
                info="Select or type a model name",
            )
            refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm", visible=(user_prefs.provider == "ollama"))
            model_info_html = gr.HTML(
                '<p style="font-size: 0.85em; color: #888; margin: 4px 0 8px 0; text-align: center;">'
                'Requires a vision model (llava, qwen3-vl, etc.) – '
                '<a href="https://ollama.com/search?c=vision" target="_blank" '
                'style="color: #58a6ff; text-decoration: none;">Browse vision models ↗</a>'
                '</p>',
                visible=(user_prefs.provider == "ollama")
            )

        preset_dropdown = gr.Dropdown(
            label="Prompt Preset",
            choices=preset_choices,
            value=default_preset,
            info="Select a prompt preset for your use case. You can add or edit presets in config.yaml",
        )

        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            value=user_prefs.temperature,
            step=0.1,
            info="Higher = more creative, Lower = more deterministic",
        )

        with gr.Row():
            prefix_input = gr.Textbox(
                label="Prefix",
                value=user_prefs.prefix,
                placeholder="Text added before description...",
                info="Added at the start of each description",
                lines=2,
            )
            suffix_input = gr.Textbox(
                label="Suffix",
                value=user_prefs.suffix,
                placeholder="Text added after description...",
                info="Added at the end of each description",
                lines=2,
            )

        overwrite_checkbox = gr.Checkbox(
            label="Overwrite existing files",
            value=user_prefs.overwrite,
            info="If unchecked, skip images that already have .txt files",
        )

        # State to track if processing is running
        is_processing = gr.State(False)

        with gr.Row():
            process_btn = gr.Button("Start Processing", variant="primary", size="lg", scale=3)
            stop_btn = gr.Button("Stop", variant="stop", size="lg", scale=1, interactive=False)

        status_text = gr.Textbox(
            label="Processing Status",
            value="Ready",
            interactive=False,
        )



        def toggle_provider(provider: str):
            """Toggle between Ollama and OpenAI provider UI."""
            if provider == "ollama":
                return (
                    gr.update(visible=True),   # ollama_group
                    gr.update(visible=False),  # openai_group
                    gr.update(visible=True),   # refresh_models_btn
                    gr.update(visible=True),   # model_info_html
                    "<span style='color: gray;'>[Not Connected]</span>",  # connection_status
                )
            else:  # openai
                return (
                    gr.update(visible=False),  # ollama_group
                    gr.update(visible=True),   # openai_group
                    gr.update(visible=False),  # refresh_models_btn
                    gr.update(visible=False),  # model_info_html
                    "<span style='color: gray;'>[Not Connected]</span>",  # connection_status
                )

        def check_ollama_connection(host: str):
            """Check connection to Ollama and refresh models list."""
            if not host:
                return "<span style='color: red;'>[Disconnected]</span>", gr.update(choices=[], value=None)
            
            success, message = test_ollama_connection(host)
            if success:
                models = get_ollama_models(host)
                # Use saved model if available and valid
                saved_model = user_prefs.ollama_model
                default_model = saved_model if saved_model in models else (models[0] if models else None)
                return f"<span style='color: #00ff00;'>{message}</span>", gr.update(choices=models, value=default_model)
            else:
                return f"<span style='color: red;'>{message}</span>", gr.update(choices=[], value=None)

        def check_openai_connection(base_url: str, api_key: str):
            """Check connection to OpenAI-compatible API and load models."""
            if not api_key:
                return "<span style='color: red;'>✗ API key required</span>", gr.update(choices=[], value=None)
            
            success, message = test_connection(
                provider="openai",
                openai_base_url=base_url or None,
                openai_api_key=api_key,
            )
            
            if success:
                try:
                    models = list_models(
                        provider="openai",
                        openai_base_url=base_url or None,
                        openai_api_key=api_key,
                    )
                    # Use saved model if available and valid
                    saved_model = user_prefs.openai_model
                    default_model = saved_model if saved_model in models else (models[0] if models else None)
                    return f"<span style='color: #00ff00;'>{message}</span>", gr.update(choices=models, value=default_model)
                except Exception as e:
                    return f"<span style='color: orange;'>✓ Connected but failed to list models: {str(e)[:30]}</span>", gr.update(choices=[], value=None)
            else:
                return f"<span style='color: red;'>{message}</span>", gr.update(choices=[], value=None)

        def refresh_ollama_models(host: str):
            """Refresh models list from Ollama."""
            models = get_ollama_models(host)
            return gr.update(choices=models, value=models[0] if models else None)

        def format_time(seconds: float) -> str:
            """Format seconds into human readable time."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"

        def stop_processing():
            """Stop the current processing immediately."""
            _stop_event.set()
            return (
                "Stopping... (will stop after current image)",
                gr.update(interactive=True),  # Re-enable start button
                gr.update(interactive=False, value="Stopping..."),  # Disable stop button
                False,
            )

        def process_folder(
            folder_path: str,
            provider: str,
            ollama_host: str,
            openai_base_url: str,
            openai_api_key: str,
            model: str,
            preset_key: str,
            temperature: float,
            prefix: str,
            suffix: str,
            overwrite: bool,
            processing: bool,
            progress=gr.Progress(),
        ):
            # Clear any previous stop signal
            _stop_event.clear()

            if not folder_path:
                yield (
                    "Error: Please enter a folder path",
                    gr.update(interactive=True),  # process_btn
                    gr.update(interactive=False, value="Stop"),  # stop_btn
                    False,
                )
                return

            folder = Path(folder_path)
            if not folder.exists():
                yield (
                    f"Error: Folder not found: {folder_path}",
                    gr.update(interactive=True),
                    gr.update(interactive=False, value="Stop"),
                    False,
                )
                return

            if not folder.is_dir():
                yield (
                    f"Error: Path is not a folder: {folder_path}",
                    gr.update(interactive=True),
                    gr.update(interactive=False, value="Stop"),
                    False,
                )
                return

            system_prompt = preset_prompts.get(preset_key, "Describe this image.")
            markdown_format = preset_markdown_format.get(preset_key, False)

            # Validate provider-specific requirements
            if provider == "openai" and not openai_api_key:
                yield (
                    "Error: API key is required for OpenAI-compatible provider",
                    gr.update(interactive=True),
                    gr.update(interactive=False, value="Stop"),
                    False,
                )
                return

            # Save user preferences for next session (excluding sensitive data)
            save_user_preferences(UserPreferences(
                image_folder=folder_path,
                provider=provider,
                ollama_model=model if provider == "ollama" else "",
                openai_model=model if provider == "openai" else "",
                preset_key=preset_key or "",
                prefix=prefix or "",
                suffix=suffix or "",
                temperature=temperature,
                overwrite=overwrite,
            ))

            config = Config(
                provider=provider,
                ollama_host=ollama_host or get_ollama_host(),
                openai_base_url=openai_base_url or None,
                openai_api_key=openai_api_key or None,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                description_prefix=prefix,
                description_suffix=suffix,
                markdown_format=markdown_format,
            )

            images = find_images(folder, config.supported_extensions)
            if not images:
                yield (
                    "No images found in folder",
                    gr.update(interactive=True),
                    gr.update(interactive=False, value="Stop"),
                    False,
                )
                return

            # Yield initial state: disable start, enable stop
            yield (
                f"Starting... Found {len(images)} images",
                gr.update(interactive=False),  # Disable start button
                gr.update(interactive=True, value="Stop"),  # Enable stop button
                True,
            )

            processed_count = 0
            skipped_count = 0

            # Sliding window for time estimation (last 5 images)
            time_window: deque[float] = deque(maxlen=5)
            last_time = time.time()

            # Use tqdm-style progress bar
            generator = process_images_generator(folder, config, overwrite, _stop_event)

            for current, total, image_path, description, is_error in progress.tqdm(
                generator, total=len(images), desc="Processing"
            ):
                # Check if stop was requested
                if _stop_event.is_set():
                    yield (
                        f"Stopped! {processed_count} processed, {skipped_count} skipped",
                        gr.update(interactive=True),  # Re-enable start
                        gr.update(interactive=False, value="Stop"),  # Disable stop
                        False,
                    )
                    return

                # Calculate time for this image
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time

                if not is_error:
                    time_window.append(elapsed)
                    processed_count += 1
                else:
                    skipped_count += 1

                # Calculate ETA
                remaining = total - current
                if time_window and remaining > 0:
                    avg_time = sum(time_window) / len(time_window)
                    eta = avg_time * remaining
                    eta_str = f" - ETA: {format_time(eta)}"
                else:
                    eta_str = ""

                status = f"Processing: {current}/{total} ({image_path.name}){eta_str}"
                yield (
                    status,
                    gr.update(interactive=False),  # Keep start disabled
                    gr.update(interactive=True, value="Stop"),  # Keep stop enabled
                    True,
                )

            yield (
                f"Done! {processed_count} processed, {skipped_count} skipped",
                gr.update(interactive=True),  # Re-enable start
                gr.update(interactive=False, value="Stop"),  # Disable stop
                False,
            )

        # Provider toggle
        provider_radio.change(
            fn=toggle_provider,
            inputs=[provider_radio],
            outputs=[ollama_group, openai_group, refresh_models_btn, model_info_html, connection_status],
        )

        # Ollama connection
        connect_btn.click(
            fn=check_ollama_connection,
            inputs=[ollama_host_input],
            outputs=[connection_status, model_dropdown],
        )

        refresh_models_btn.click(
            fn=refresh_ollama_models,
            inputs=[ollama_host_input],
            outputs=[model_dropdown],
        )

        # OpenAI connection
        connect_openai_btn.click(
            fn=check_openai_connection,
            inputs=[openai_base_url_input, openai_api_key_input],
            outputs=[connection_status, model_dropdown],
        )

        # Auto-connect on load (only for Ollama)
        app.load(
            fn=lambda provider, host: check_ollama_connection(host) if provider == "ollama" else ("<span style='color: gray;'>[Not Connected]</span>", gr.update()),
            inputs=[provider_radio, ollama_host_input],
            outputs=[connection_status, model_dropdown],
        )

        process_btn.click(
            fn=process_folder,
            inputs=[
                folder_input,
                provider_radio,
                ollama_host_input,
                openai_base_url_input,
                openai_api_key_input,
                model_dropdown,
                preset_dropdown,
                temperature_slider,
                prefix_input,
                suffix_input,
                overwrite_checkbox,
                is_processing,
            ],
            outputs=[status_text, process_btn, stop_btn, is_processing],
        )

        stop_btn.click(
            fn=stop_processing,
            inputs=[],
            outputs=[status_text, process_btn, stop_btn, is_processing],
        )

        gr.HTML(
            """
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #444; text-align: center; color: #888; font-size: 0.9em;">
                <p>
                    <a href="https://github.com/hydropix/ollama-image-describer" target="_blank" style="color: #58a6ff; text-decoration: none;">
                        GitHub Project
                    </a>
                </p>
                <p style="margin-top: 8px;">
                    Found a bug? Have a suggestion?
                    <a href="https://github.com/hydropix/ollama-image-describer/issues" target="_blank" style="color: #58a6ff; text-decoration: none;">
                        Open an issue
                    </a>
                    or give a star to support the project!
                </p>
            </div>
            """
        )

    return app


def launch_web_app():
    """Launch the Gradio web application."""
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=["/", "C:\\", "D:\\", "E:\\", "Z:\\"],
    )


if __name__ == "__main__":
    launch_web_app()
