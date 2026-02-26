# 3DMODEL - Chadpocalypse TRELLIS.2 Pipeline

Image-to-3D mesh generation API powered by Microsoft TRELLIS.2-4B on RunPod.

## Architecture

- **Docker Image**: `camenduru/tostui-trellis2` (pre-built CUDA extensions + model weights)
- **API Server**: FastAPI on port 8000
- **GPU**: NVIDIA A40 (48GB VRAM)

## Presets

| Preset | Faces | Texture | Remesh | Use Case |
|--------|-------|---------|--------|----------|
| `lowpoly` | 5,000 | 1024 | Yes | Game-ready characters |
| `default` | 1,000,000 | 2048 | No | Full quality output |

## API Endpoints

- `GET /health` — Check if pipeline is loaded
- `GET /presets` — View available presets and their settings
- `POST /generate` — Generate 3D model from image
- `GET /download/{filename}` — Download generated GLB
- `GET /outputs` — List all generated files

## RunPod Template

| Field | Value |
|-------|-------|
| Template ID | `eixrhp7kx7` |
| Container Image | `camenduru/tostui-trellis2` |
| HTTP Ports | 3000, 8000 |
| TCP Ports | 22 |

## PowerShell Usage

```powershell
# Generate with lowpoly preset
$body = @{
    image_url = "https://your-image-url.png"
    preset = "lowpoly"
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://<pod-url>:8000/generate" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body
```

## Files

- `trellis_server.py` — FastAPI server with preset system
- `start.sh` — Pod boot script
