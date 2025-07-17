# AI Video Generator

This repository collects a set of small demos for various open source image-to-video models.  A simple Streamlit
front-end is provided together with a FastAPI back-end used for WebSocket communication.

The `external/` directory contains the upstream model repositories.  Model weights are downloaded using
`huggingface_hub` via the helper scripts in `scripts/`.

## Directory structure

```
app/         FastAPI application and Streamlit demo
external/    Third party model repositories
fronted/     Additional Streamlit example for real-time processing
processing/  Utility pipelines used by the back-end
scripts/     Helper scripts for downloading model weights
output/      Example output video
```

## Quick start

1. Install the required Python packages (see `requirements.txt`). Some models
   require GPU support and additional packages which are downloaded with
   `scripts/setup_repo.sh`.
2. Run the Streamlit demo:

```bash
streamlit run app/quick_demo.py
```

3. Upload an image, choose the desired model and generate a short video clip.

The FastAPI server can also be started directly:

```bash
python app/main.py
```

A minimal GPU check is available in `test/test.sh`.
