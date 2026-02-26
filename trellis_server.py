#!/usr/bin/env python3
"""
Chadpocalypse TRELLIS.2 API Server
FastAPI REST API for image-to-3D mesh generation.
Port 8000 | Docs at /docs

Presets:
  - default: Vanilla TRELLIS.2 settings (1M faces, 2048 texture)
  - lowpoly: Optimized for game-ready low-poly characters (5k faces, 1024 texture)
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "autotune_cache.json"
)
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "1"

import io
import uuid
import shutil
import tempfile
import logging
from datetime import datetime
from typing import Optional

import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# TRELLIS.2 imports
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# ─── Logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trellis-api")

# ─── Output directory ───
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Presets ───
PRESETS = {
    "default": {
        "resolution": "1024",
        "decimation_target": 1000000,
        "texture_size": 2048,
        "remesh": False,
        "remesh_band": 1,
        "remesh_project": 0.9,
        "mesh_cluster_refine_iterations": 0,
        "mesh_cluster_global_iterations": 1,
        "mesh_cluster_smooth_strength": 1,
        "ss_guidance_strength": 7.5,
        "ss_guidance_rescale": 0.7,
        "ss_sampling_steps": 12,
        "ss_rescale_t": 5.0,
        "shape_slat_guidance_strength": 7.5,
        "shape_slat_guidance_rescale": 0.5,
        "shape_slat_sampling_steps": 12,
        "shape_slat_rescale_t": 3.0,
        "tex_slat_guidance_strength": 1.0,
        "tex_slat_guidance_rescale": 0.0,
        "tex_slat_sampling_steps": 12,
        "tex_slat_rescale_t": 3.0,
    },
    "lowpoly": {
        "resolution": "1024",
        "decimation_target": 5000,
        "texture_size": 1024,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0.7,
        "mesh_cluster_refine_iterations": 2,
        "mesh_cluster_global_iterations": 1,
        "mesh_cluster_smooth_strength": 1,
        "ss_guidance_strength": 7.5,
        "ss_guidance_rescale": 0.7,
        "ss_sampling_steps": 12,
        "ss_rescale_t": 5.0,
        "shape_slat_guidance_strength": 7.5,
        "shape_slat_guidance_rescale": 0.5,
        "shape_slat_sampling_steps": 12,
        "shape_slat_rescale_t": 3.0,
        "tex_slat_guidance_strength": 1.0,
        "tex_slat_guidance_rescale": 0.0,
        "tex_slat_sampling_steps": 12,
        "tex_slat_rescale_t": 3.0,
    },
}


# ─── Request model ───
class GenerateRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the input image")
    image_base64: Optional[str] = Field(None, description="Base64-encoded input image")
    preset: str = Field("lowpoly", description="Preset name: 'default' or 'lowpoly'")
    seed: Optional[int] = Field(None, description="Random seed (None = random)")
    # Optional overrides (if set, override the preset value)
    resolution: Optional[str] = Field(None, description="Override: 512, 1024, or 1536")
    decimation_target: Optional[int] = Field(None, description="Override: target face count")
    texture_size: Optional[int] = Field(None, description="Override: texture resolution")
    remesh: Optional[bool] = Field(None, description="Override: enable remeshing")
    filename: Optional[str] = Field(None, description="Custom output filename (without extension)")


# ─── App ───
app = FastAPI(title="Chadpocalypse TRELLIS.2 API", version="1.0.0")

# Global pipeline reference
pipeline = None


def load_pipeline():
    """Load the TRELLIS.2 pipeline into GPU memory."""
    global pipeline
    if pipeline is not None:
        return
    logger.info("Loading TRELLIS.2-4B pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.rembg_model = None
    pipeline.low_vram = False
    pipeline.cuda()
    logger.info("Pipeline loaded and ready.")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess: resize, crop to bounding box of alpha, center."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Resize if too large
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    # Crop to alpha bounding box
    alpha = np.array(image)[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        return image
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    crop_box = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(crop_box)

    # Premultiply alpha
    output = np.array(image).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))


@app.on_event("startup")
async def startup():
    load_pipeline()


@app.get("/health")
async def health():
    return {
        "status": "ready" if pipeline is not None else "loading",
        "model": "TRELLIS.2-4B",
        "presets": list(PRESETS.keys()),
    }


@app.get("/presets")
async def get_presets():
    return PRESETS


@app.post("/generate")
async def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    # Validate input
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    # Load preset
    if req.preset not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{req.preset}'. Available: {list(PRESETS.keys())}",
        )
    params = PRESETS[req.preset].copy()

    # Apply overrides
    if req.resolution is not None:
        params["resolution"] = req.resolution
    if req.decimation_target is not None:
        params["decimation_target"] = req.decimation_target
    if req.texture_size is not None:
        params["texture_size"] = req.texture_size
    if req.remesh is not None:
        params["remesh"] = req.remesh

    # Seed
    seed = req.seed if req.seed is not None else np.random.randint(0, np.iinfo(np.int32).max)

    # Load image
    try:
        if req.image_base64:
            import base64
            image_data = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        else:
            import urllib.request
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                urllib.request.urlretrieve(req.image_url, tmp.name)
                image = Image.open(tmp.name).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    # Preprocess
    image = preprocess_image(image)

    # Generate
    logger.info(f"Generating with preset='{req.preset}', seed={seed}, resolution={params['resolution']}, faces={params['decimation_target']}")
    try:
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": params["ss_sampling_steps"],
                "guidance_strength": params["ss_guidance_strength"],
                "guidance_rescale": params["ss_guidance_rescale"],
                "rescale_t": params["ss_rescale_t"],
            },
            shape_slat_sampler_params={
                "steps": params["shape_slat_sampling_steps"],
                "guidance_strength": params["shape_slat_guidance_strength"],
                "guidance_rescale": params["shape_slat_guidance_rescale"],
                "rescale_t": params["shape_slat_rescale_t"],
            },
            tex_slat_sampler_params={
                "steps": params["tex_slat_sampling_steps"],
                "guidance_strength": params["tex_slat_guidance_strength"],
                "guidance_rescale": params["tex_slat_guidance_rescale"],
                "rescale_t": params["tex_slat_rescale_t"],
            },
            pipeline_type={
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[params["resolution"]],
            return_latent=True,
        )

        mesh = outputs[0]
        mesh.simplify(16777216)  # nvdiffrast limit

        # Decode latents for GLB export
        from trellis2.modules.sparse import SparseTensor
        shape_slat, tex_slat, res = latents
        mesh_for_export = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        mesh_for_export.simplify(16777216)

        # Export to GLB
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_for_export.vertices,
            faces=mesh_for_export.faces,
            attr_volume=mesh_for_export.attrs,
            coords=mesh_for_export.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=params["decimation_target"],
            texture_size=params["texture_size"],
            remesh=params["remesh"],
            remesh_band=params["remesh_band"],
            remesh_project=params["remesh_project"],
            mesh_cluster_refine_iterations=params["mesh_cluster_refine_iterations"],
            mesh_cluster_global_iterations=params["mesh_cluster_global_iterations"],
            mesh_cluster_smooth_strength=params["mesh_cluster_smooth_strength"],
            use_tqdm=True,
        )

        # Save
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = req.filename or f"chad_{timestamp}_s{seed}"
        glb_path = os.path.join(OUTPUT_DIR, f"{filename}.glb")
        glb.export(glb_path, extension_webp=True)

        torch.cuda.empty_cache()

        logger.info(f"Generated: {glb_path}")
        return {
            "status": "success",
            "file": glb_path,
            "filename": f"{filename}.glb",
            "download_url": f"/download/{filename}.glb",
            "seed": seed,
            "preset": req.preset,
            "params": params,
        }

    except Exception as e:
        torch.cuda.empty_cache()
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.get("/download/{filename}")
async def download(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="model/gltf-binary", filename=filename)


@app.get("/outputs")
async def list_outputs():
    """List all generated GLB files."""
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".glb"):
            path = os.path.join(OUTPUT_DIR, f)
            files.append({
                "filename": f,
                "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
                "download_url": f"/download/{f}",
            })
    return {"files": files}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
