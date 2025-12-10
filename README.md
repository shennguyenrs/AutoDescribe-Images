<p align="center">
  <img src="assets/logo.png" width="400" height="400" alt="ollama-image-describer logo">
</p>

Tool to automatically generate text descriptions (captions) for images using AI vision models. Supports both **local Ollama models** (LLaVA, Qwen3-VL, Llama Vision) and **OpenAI-compatible APIs** (OpenAI, OpenRouter, Azure, etc.). Available as a **web application** (recommended) or **CLI**.

**Key features:**
- 🔄 **Dual Provider Support**: Use local Ollama models or cloud-based OpenAI-compatible APIs
- 🎨 **Fully Customizable Prompts**: Control output format with detailed system prompts
- 📋 **Built-in Presets**: Optimized for Stable Diffusion, Z-Image, Flux, and more

## Use Case

**Perfect for AI image generation training!** This tool is designed to help you create caption files for training LoRA (Low-Rank Adaptation) models on image generation AI like Stable Diffusion, Flux, Z-Image, or other diffusion models.

When training a LoRA, each image in your dataset needs an accompanying `.txt` file with a description. This tool automates that process by:
- Analysis of each image using a visual AI model, with your personalized instructions in natural language
- Generating detailed, consistent descriptions tailored to your target model

## Prerequisites

- **[Git](https://git-scm.com/)** - Version control
- **[uv](https://docs.astral.sh/uv/)** - Python package manager (handles Python installation automatically)
- **[Ollama](https://ollama.com/download)** (Optional) - Required only if using local Ollama models
- **OpenAI-compatible API** (Optional) - Required only if using cloud-based providers (OpenAI, OpenRouter, etc.)

### Installing Git

Check if Git is already installed:

```bash
git --version
```

If not installed, download it from [git-scm.com](https://git-scm.com/downloads) and follow the installation instructions for your OS.

### Installing uv

**uv** is a fast Python package manager. First, check if you already have it installed:

```bash
uv --version
```

If not installed, run one of these commands:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal to ensure `uv` is available in your PATH.

### Installing Ollama (For Local Models)

**Only required if you plan to use local Ollama models.** Skip this if you're using OpenAI-compatible APIs.

Download and install Ollama from [ollama.com](https://ollama.com/download), then pull a vision model:

```bash
ollama pull qwen3-vl:4b
```

### Getting API Access (For Cloud Providers)

**Only required if you plan to use OpenAI-compatible APIs.**

#### OpenAI
1. Sign up at [platform.openai.com](https://platform.openai.com/)
2. Get your API key from [API Keys page](https://platform.openai.com/api-keys)

#### OpenRouter
1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Get your API key from [Keys page](https://openrouter.ai/keys)
3. Browse available models at [openrouter.ai/models](https://openrouter.ai/models)

#### Azure OpenAI
1. Set up Azure OpenAI service in Azure Portal
2. Get your endpoint and API key
3. Use your deployment endpoint as base URL

## Installation

```bash
# Clone the repository
git clone https://github.com/hydropix/ollama-image-describer.git
cd ollama-image-describer

# Install Python dependencies with uv
uv sync
```

> **Note:** `uv sync` installs the Python dependencies (including the `ollama` Python client library). The Ollama server itself must be installed separately as described above.

## Configuration

### Environment Variables (.env)

Create a `.env` file from the example:

```bash
cp config/.env.example .env
```

#### For Ollama (Local Models)

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:4b
```

#### For OpenAI-Compatible APIs

```env
# OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5

# OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...
OPENAI_MODEL=openai/gpt-5

# Azure OpenAI
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-azure-key
OPENAI_MODEL=your-deployment-name
```

### Prompt Configuration (config/config.yaml)

The `config/config.yaml` file contains prompt presets and default settings. **The system prompt is fully customizable**, allowing you to precisely control the output format to match your specific needs.

#### Built-in Presets

The tool includes presets optimized for different image generation models:

| Preset | Target Model | Description |
|--------|--------------|-------------|
| **Z-Image** | Z-Image | Very detailed, structured descriptions with markdown formatting. Focuses on composition, lighting, textures, and atmosphere with poetic precision. |
| **Stable Diffusion** | SD, SDXL, Forge | Tag-based prompts with weight syntax `(element:1.2)`. Uses parentheses for emphasis and quality boosters like `(masterpiece:1.2), (best quality)`. |
| **Simple** | General use | Concise, straightforward descriptions without special formatting. |

#### Example: Stable Diffusion Preset Output

```
(masterpiece:1.2), (best quality), 1girl, long flowing red hair, (emerald green eyes:1.3),
elegant black dress, standing in flower field, soft golden hour lighting, (bokeh:1.1),
depth of field, vibrant colors, digital painting style, highly detailed
```

#### Example: Z-Image Preset Output

```markdown
## Subject
**Young woman** with flowing auburn hair, wearing a vintage emerald dress

## Composition & Setting
Wide shot capturing a sunlit meadow with wildflowers in the foreground

## Lighting & Atmosphere
*Golden hour lighting* casting warm shadows, *soft diffused glow* from the setting sun
```

#### Creating Custom Presets

Add your own presets in `config/config.yaml` to tailor outputs for your specific workflow:

```yaml
prompts:
  # Your custom preset
  flux:
    name: "Flux"
    markdownFormat: false
    prompt: |
      Generate a natural language description optimized for Flux models.
      Focus on clear, descriptive sentences without weight syntax.
      Describe the scene as if telling a story...

  my_custom:
    name: "My Custom Style"
    markdownFormat: false
    prompt: |
      Your custom instructions here...
      Be specific about the output format you want.

defaults:
  provider: "ollama"  # or "openai"
  temperature: 0.7
  model: "qwen3-vl:4b"  # Ollama model
  # max_tokens: 8192    # OpenAI setting
```

The `markdownFormat` option controls whether the output uses markdown styling (headers, bold, italics) or plain text.

## Usage

> **Important:** All commands must be run from the project directory (`ollama-image-describer`).

### Web Interface (Recommended)

Launch the web interface for an easier experience:

```bash
cd ollama-image-describer
uv run python -m image_describer --web
```

#### Using the Web Interface

1. **Select Provider**
   - Choose **Ollama (Local)** for local models
   - Choose **OpenAI-Compatible API** for cloud providers

2. **Connect to Provider**
   - **Ollama**: Click "Test Connection" (auto-connects to localhost)
   - **OpenAI**: Enter Base URL and API Key, then click "Test Connection & Load Models"

3. **Select Model**
   - Models are automatically loaded after successful connection
   - For OpenAI: Popular models include `gpt-4o`, `gpt-4-turbo`, `claude-3-opus` (via OpenRouter)

4. **Choose Settings**
   - Select a preset (Stable Diffusion, Z-Image, Simple)
   - Adjust temperature, add prefix/suffix if needed

5. **Process Images**
   - Browse to your image folder
   - Click "Start Processing"

### CLI Mode

```bash
uv run python -m image_describer <image_folder> [options]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--web` | `-w` | Launch the web interface |
| `--provider` | | Vision provider: `ollama` or `openai` (default: ollama) |
| `--openai-url` | | OpenAI-compatible API base URL |
| `--openai-key` | | OpenAI-compatible API key |
| `--config` | `-c` | Path to YAML config file |
| `--prefix` | `-p` | Text prepended to each description |
| `--suffix` | `-s` | Text appended to each description (e.g., ", By Artist") |
| `--overwrite/--no-overwrite` | | Overwrite existing .txt files (default: overwrite) |
| `--verbose` | `-v` | Verbose mode |
| `--debug` | `-d` | Enable debug logging |

### Examples

```bash
# Launch web interface (easiest)
uv run python -m image_describer --web

# Using Ollama (default)
uv run python -m image_describer ./my_images

# Using OpenAI
uv run python -m image_describer ./my_images \
  --provider openai \
  --openai-url https://api.openai.com/v1 \
  --openai-key sk-...

# Using OpenRouter
uv run python -m image_describer ./my_images \
  --provider openai \
  --openai-url https://openrouter.ai/api/v1 \
  --openai-key sk-or-v1-...

# With prefix and suffix
uv run python -m image_describer ./my_images --prefix "A photo of " --suffix ", By Kristof"

# Using environment variables (recommended for API keys)
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
uv run python -m image_describer ./my_images --provider openai

# Verbose mode with custom config
uv run python -m image_describer ./my_images -v -c custom_config.yaml
```

## Supported Models

> ⚠️ **Important:** This tool requires a **vision model** capable of analyzing images. Standard text-only models (like `llama3`, `mistral`, etc.) will not work.
>
> Browse all available vision models: [Ollama Vision Models](https://ollama.com/search?c=vision)

**Recommended vision models:**

- **Qwen3-VL** (recommended): `qwen3-vl:4b`
- **LLaVA**: `llava`, `llava:13b`
- **Llama Vision**: `llama3.2-vision`

## License

MIT
