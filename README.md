# CAD Quoting Tool

## Setup

1. Install Python 3.11 or newer on the target machine.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. Install the runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the application

After installing the dependencies, run the main application entry point:

```bash
python appV5.py
```

You can quickly verify the environment configuration without launching the UI:

```bash
python appV5.py --print-env
```

Refer to the [deployment guide](docs/deployment_guide.md) for end-to-end packaging and
host preparation steps when moving the tool to another device.
