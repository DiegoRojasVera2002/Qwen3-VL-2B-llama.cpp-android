from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import base64, io, numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolo11n-seg.pt")  # descarga ~6MB automático

COLORS = [
    (255, 80,  80 ), (80,  220, 80 ), (80,  80,  255),
    (255, 220, 50 ), (200, 80,  255), (50,  220, 255),
    (255, 140, 50 ), (50,  255, 180), (255, 80,  180),
    (140, 255, 50 ), (50,  140, 255), (255, 180, 80 ),
]

class ImageRequest(BaseModel):
    image: str  # base64 JPEG

@app.post("/segment")
async def segment(req: ImageRequest):
    img_data = base64.b64decode(req.image)
    img      = Image.open(io.BytesIO(img_data)).convert("RGB")
    w, h     = img.size
    img_np   = np.array(img)

    results = model(img_np, device="cpu", verbose=False)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            r, g, b      = COLORS[i % len(COLORS)]
            mask_img     = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = np.array(mask_img.resize((w, h), Image.BILINEAR))
            px = mask_resized > 128
            overlay[px, 0] = r
            overlay[px, 1] = g
            overlay[px, 2] = b
            overlay[px, 3] = 140

    buf = io.BytesIO()
    Image.fromarray(overlay, "RGBA").save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    print("Calentando modelo...")
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
    print("YOLO server listo → http://localhost:8081")
    uvicorn.run(app, host="0.0.0.0", port=8081)
