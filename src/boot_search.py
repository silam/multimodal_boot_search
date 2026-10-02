"""
Boot Catalog — Multi-Image Qdrant Ingestion + MaxSim Query
==========================================================
Strategy: each image = one Qdrant point, linked by SKU in payload.
At query time, MaxSim aggregates scores across all points for a SKU,
then we deduplicate and rank by SKU-level score.
"""

import uuid
import asyncio
from io import BytesIO
from typing import Optional

import numpy as np
import httpx
import open_clip
import torch
from PIL import Image
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    NamedVector,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"
COLLECTION  = "boots_catalog_multi"
CLIP_MODEL  = "ViT-L-14"
CLIP_PRETRAINED = "openai"
IMAGE_VECTOR_SIZE = 768
TOP_K_POINTS = 50   # fetch this many raw points, then MaxSim-aggregate


# ─────────────────────────────────────────────
# CLIP SETUP
# ─────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAINED
)
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)


def embed_image(pil_image: Image.Image) -> list[float]:
    """CLIP-encode a PIL image → normalized 768-dim vector."""
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = clip_model.encode_image(tensor)
        vec = vec / vec.norm(dim=-1, keepdim=True)   # L2-normalize
    return vec.squeeze(0).cpu().tolist()


def embed_text(text: str) -> list[float]:
    """CLIP-encode a text string → normalized 768-dim vector."""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        vec = clip_model.encode_text(tokens)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).cpu().tolist()


async def fetch_image(url: str, client: httpx.AsyncClient) -> Image.Image:
    resp = await client.get(url, timeout=15.0)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ─────────────────────────────────────────────
# COLLECTION SETUP
# ─────────────────────────────────────────────

async def create_collection(qdrant: AsyncQdrantClient):
    """
    Create collection if it doesn't exist.
    One named vector: 'image' (768-dim CLIP).
    Each point = one image of one boot SKU.
    """
    existing = await qdrant.get_collections()
    names = [c.name for c in existing.collections]
    if COLLECTION in names:
        print(f"Collection '{COLLECTION}' already exists — skipping creation.")
        return

    await qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "image": VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
        },
    )

    # Payload indexes for fast filtering
    await qdrant.create_payload_index(COLLECTION, "sku",        PayloadSchemaType.KEYWORD)
    await qdrant.create_payload_index(COLLECTION, "model_name", PayloadSchemaType.KEYWORD)
    await qdrant.create_payload_index(COLLECTION, "category",   PayloadSchemaType.KEYWORD)
    await qdrant.create_payload_index(COLLECTION, "in_stock",   PayloadSchemaType.BOOL)
    await qdrant.create_payload_index(COLLECTION, "image_index",PayloadSchemaType.INTEGER)

    print(f"Collection '{COLLECTION}' created.")


# ─────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────

async def ingest_boot(boot: dict, qdrant: AsyncQdrantClient):
    """
    Ingest one boot record.

    boot = {
        "sku":         "8111",
        "model_name":  "Iron Ranger",
        "description": "Rugged cap-toe boot...",
        "features":    ["steel toe", "waterproof"],
        "category":    "work",
        "price_usd":   389,
        "in_stock":    True,
        "color":       "amber harness",
        "image_urls":  ["https://...", "https://...", "https://..."],
    }

    Each image_url → one separate Qdrant point.
    Point ID is deterministic: uuid5(sku + image_index) → stable, re-runnable.
    """
    sku         = boot["sku"]
    image_urls  = boot["image_urls"]
    points      = []

    async with httpx.AsyncClient() as http:
        for idx, url in enumerate(image_urls):
            try:
                pil_img = await fetch_image(url, http)
            except Exception as e:
                print(f"  [WARN] SKU {sku} image {idx} failed: {e}")
                continue

            vec = embed_image(pil_img)

            # Deterministic point ID so re-ingestion is idempotent
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sku}::img{idx}"))

            points.append(PointStruct(
                id=point_id,
                vectors={"image": vec},
                payload={
                    # ── Boot metadata (same on every point for this SKU) ──
                    "sku":          sku,
                    "model_name":   boot["model_name"],
                    "description":  boot["description"],
                    "features":     boot["features"],
                    "category":     boot["category"],
                    "price_usd":    boot["price_usd"],
                    "in_stock":     boot["in_stock"],
                    "color":        boot.get("color", ""),
                    # ── Per-image metadata ──
                    "image_index":  idx,          # 0 = hero shot
                    "image_url":    url,
                    "total_images": len(image_urls),
                },
            ))
            print(f"  Embedded SKU {sku} image {idx}/{len(image_urls)-1}")

    if points:
        await qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"  ✓ Upserted {len(points)} points for SKU {sku}")
    else:
        print(f"  ✗ No valid images for SKU {sku}")


async def ingest_catalog(catalog: list[dict]):
    """Ingest full catalog. Processes boots concurrently in batches of 5."""
    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    await create_collection(qdrant)

    # Process in batches to avoid overwhelming image servers
    batch_size = 5
    for i in range(0, len(catalog), batch_size):
        batch = catalog[i : i + batch_size]
        print(f"\nIngesting batch {i//batch_size + 1} ({len(batch)} boots)...")
        await asyncio.gather(*[ingest_boot(boot, qdrant) for boot in batch])

    await qdrant.close()
    print("\n✓ Catalog ingestion complete.")


# ─────────────────────────────────────────────
# MAXSIM AGGREGATION
# ─────────────────────────────────────────────

def maxsim_aggregate(raw_results: list) -> list[dict]:
    """
    MaxSim: for each SKU, keep the HIGHEST similarity score
    across all its image points. Then sort SKUs by that score.

    raw_results: list of ScoredPoint from qdrant search
    returns: list of dicts sorted by sku_score desc, deduplicated by SKU
    """
    sku_best: dict[str, dict] = {}

    for point in raw_results:
        sku   = point.payload["sku"]
        score = point.score

        if sku not in sku_best or score > sku_best[sku]["score"]:
            sku_best[sku] = {
                "sku":           sku,
                "score":         score,
                "model_name":    point.payload["model_name"],
                "description":   point.payload["description"],
                "features":      point.payload["features"],
                "category":      point.payload["category"],
                "price_usd":     point.payload["price_usd"],
                "in_stock":      point.payload["in_stock"],
                "color":         point.payload["color"],
                "matched_image": point.payload["image_url"],   # best-matching image
                "image_index":   point.payload["image_index"],
            }

    # Sort by best-matching image score descending
    return sorted(sku_best.values(), key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────
# QUERY FUNCTIONS
# ─────────────────────────────────────────────

async def search_by_image(
    query_image: Image.Image,
    top_n: int = 10,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Query with an image. Returns top_n unique boots ranked by MaxSim.

    filters (optional): {
        "category": "work",
        "in_stock": True,
    }
    """
    qdrant    = AsyncQdrantClient(url=QDRANT_URL)
    query_vec = embed_image(query_image)

    qdrant_filter = _build_filter(filters)

    # Fetch more raw points than needed — MaxSim will collapse to unique SKUs
    raw = await qdrant.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name="image", vector=query_vec),
        query_filter=qdrant_filter,
        limit=TOP_K_POINTS,
        with_payload=True,
        score_threshold=0.0,
    )

    results = maxsim_aggregate(raw)[:top_n]

    await qdrant.close()
    return results


async def search_by_image_and_text(
    query_image: Image.Image,
    query_text: str,
    top_n: int = 10,
    alpha: float = 0.6,        # 0.0 = text only, 1.0 = image only
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Multimodal query: image + text, fused in CLIP's joint embedding space.
    alpha controls image vs text weight.
    """
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    img_vec = np.array(embed_image(query_image))
    txt_vec = np.array(embed_text(query_text))

    # Weighted fusion, re-normalize
    fused = alpha * img_vec + (1 - alpha) * txt_vec
    fused = fused / np.linalg.norm(fused)

    qdrant_filter = _build_filter(filters)

    raw = await qdrant.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name="image", vector=fused.tolist()),
        query_filter=qdrant_filter,
        limit=TOP_K_POINTS,
        with_payload=True,
        score_threshold=0.0,
    )

    results = maxsim_aggregate(raw)[:top_n]

    await qdrant.close()
    return results


def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
    if not filters:
        return None
    conditions = []
    for key, val in filters.items():
        conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
    return Filter(must=conditions)


# ─────────────────────────────────────────────
# FASTAPI ENDPOINTS
# ─────────────────────────────────────────────

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import base64

app = FastAPI(title="Boot Visual Search API")


@app.post("/search/image")
async def api_search_image(
    image: UploadFile = File(...),
    top_n: int = Form(10),
    category: Optional[str] = Form(None),
    in_stock: Optional[bool] = Form(None),
):
    """Search by uploaded boot image."""
    pil = Image.open(BytesIO(await image.read())).convert("RGB")
    filters = {}
    if category:  filters["category"] = category
    if in_stock is not None: filters["in_stock"] = in_stock

    results = await search_by_image(pil, top_n=top_n, filters=filters or None)
    return JSONResponse({"results": results, "count": len(results)})


@app.post("/search/multimodal")
async def api_search_multimodal(
    image: UploadFile = File(...),
    text: str = Form(...),
    alpha: float = Form(0.6),
    top_n: int = Form(10),
    in_stock: Optional[bool] = Form(None),
):
    """Search by image + text feature description."""
    pil = Image.open(BytesIO(await image.read())).convert("RGB")
    filters = {}
    if in_stock is not None: filters["in_stock"] = in_stock

    results = await search_by_image_and_text(
        pil, text, top_n=top_n, alpha=alpha, filters=filters or None
    )
    return JSONResponse({"results": results, "count": len(results)})


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

SAMPLE_CATALOG = [
    {
        "sku": "8111",
        "model_name": "Iron Ranger",
        "description": "Rugged cap-toe boot built for tough terrain.",
        "features": ["steel toe", "vibram sole", "full-grain leather"],
        "category": "heritage",
        "price_usd": 389,
        "in_stock": True,
        "color": "amber harness",
        "image_urls": [
            "https://example.com/8111_front.jpg",
            "https://example.com/8111_side.jpg",
            "https://example.com/8111_sole.jpg",
            "https://example.com/8111_back.jpg",
        ],
    },
    {
        "sku": "2406",
        "model_name": "Supersole 2.0",
        "description": "Lightweight composite-toe work boot with slip resistance.",
        "features": ["composite toe", "waterproof", "slip resistant", "electrical hazard"],
        "category": "work",
        "price_usd": 249,
        "in_stock": True,
        "color": "brown",
        "image_urls": [
            "https://example.com/2406_front.jpg",
            "https://example.com/2406_angle.jpg",
        ],
    },
]

if __name__ == "__main__":
    # Ingest
    asyncio.run(ingest_catalog(SAMPLE_CATALOG))

    # Query example (replace with a real image)
    async def demo_query():
        query_img = Image.open("query_boot.jpg").convert("RGB")

        print("\n── Image-only search ──")
        results = await search_by_image(query_img, top_n=5)
        for r in results:
            print(f"  SKU {r['sku']} | {r['model_name']} | score={r['score']:.4f} | matched img: {r['matched_image']}")

        print("\n── Multimodal search (image + 'waterproof work boot') ──")
        results = await search_by_image_and_text(query_img, "waterproof work boot", alpha=0.6, top_n=5)
        for r in results:
            print(f"  SKU {r['sku']} | {r['model_name']} | score={r['score']:.4f}")

    asyncio.run(demo_query())