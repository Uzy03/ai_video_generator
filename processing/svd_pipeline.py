from __future__ import annotations

from pathlib import Path
from typing import Union, List

from PIL import Image


def generate_svd(
    input_image: Union[str, Path],
    output_path: Union[str, Path],
    frame_num: int,
    steps: int,
    guidance_scale: float,
) -> Path:
    """Run Stable Video Diffusion and save an MP4 file.

    This implementation uses diffusers and enables CPU offload to reduce
    GPU memory requirements.
    """
    from diffusers import StableVideoDiffusionPipeline
    import torch
    import tempfile
    import ffmpeg

    input_image = Path(input_image)
    output_path = Path(output_path)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()

    image = Image.open(input_image).convert("RGB")
    result = pipe(
        image=image,
        num_frames=frame_num,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )

    frames: List[Image.Image] = result.frames
    temp_dir = Path(tempfile.mkdtemp())
    for i, frame in enumerate(frames):
        frame.save(temp_dir / f"{i:04d}.png")

    (
        ffmpeg
        .input(str(temp_dir / "%04d.png"), framerate=25)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run()
    )

    for p in temp_dir.glob("*.png"):
        p.unlink()
    temp_dir.rmdir()

    return output_path