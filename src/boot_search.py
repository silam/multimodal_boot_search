"""
Boot Catalog — Multimodal Vector Search with Qdrant
====================================================
Full implementation:
  - Three named vector indexes per point (text_features, image, multimodal)
  - One Qdrant point per (sku, image_index) — multi-image support
  - MaxSim aggregation at query time to collapse image points → unique SKUs
  - Three search modes: text, image, text+image (multimodal)

Dependencies:
    pip install qdrant-client open-clip-torch torch pillow httpx openai fastapi uvicorn python-multipart
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
from openai import AsyncOpenAI

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


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

QDRANT_URL        = "http://localhost:6333"
COLLECTION        = "boots_catalog"

# Named vector keys
TEXT_VEC          = "text_features"   # 3072-dim — OpenAI text-embedding-3-large
IMAGE_VEC         = "image"           # 768-dim  — CLIP ViT-L/14 image encoder
MULTI_VEC         = "multimodal"      # 768-dim  — CLIP image+text fused

TEXT_DIM          = 3072
IMAGE_DIM         = 768

# How many raw Qdrant points to fetch before MaxSim collapses them to unique SKUs.
# Rule of thumb: top_n * avg_images_per_sku * 1.5
TOP_K_POINTS      = 50


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

# OpenCLIP — encodes both images and text in a shared 768-dim vector space
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

# OpenAI client — for text-embedding-3-large (semantic feature search)
openai_client = AsyncOpenAI()   # reads OPENAI_API_KEY from env


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def embed_text_openai(text: str) -> list[float]:
    """
    Embed a text string using OpenAI text-embedding-3-large.
    Returns a 3072-dim normalized vector.
    Used for: feature/semantic text search (TEXT_VEC index).
    """
    response = await openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding


def embed_image_clip(pil_image: Image.Image) -> list[float]:
    """
    Embed a PIL image using OpenCLIP ViT-L/14.
    Returns a 768-dim L2-normalized vector.
    Used for: visual similarity search (IMAGE_VEC index).
    """
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = clip_model.encode_image(tensor)
        vec = vec / vec.norm(dim=-1, keepdim=True)   # L2 normalize
    return vec.squeeze(0).cpu().tolist()


def embed_text_clip(text: str) -> list[float]:
    """
    Embed a text string using OpenCLIP ViT-L/14 text encoder.
    Returns a 768-dim L2-normalized vector in the SAME space as embed_image_clip.
    Used for: multimodal fusion (MULTI_VEC index).
    """
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        vec = clip_model.encode_text(tokens)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).cpu().tolist()


def fuse_image_text_clip(
    img_vec: list[float],
    text: str,
    alpha: float = 0.6,
) -> list[float]:
    """
    Fuse an image CLIP vector with a text CLIP vector.
    alpha=1.0 → image only | alpha=0.0 → text only | alpha=0.6 → default blend.
    Both vectors live in CLIP's joint embedding space, so weighted sum is valid.
    Returns a 768-dim L2-normalized vector.
    """
    iv = np.array(img_vec)
    tv = np.array(embed_text_clip(text))
    fused = alpha * iv + (1 - alpha) * tv
    fused = fused / np.linalg.norm(fused)
    return fused.tolist()


async def fetch_image_from_url(url: str, http: httpx.AsyncClient) -> Image.Image:
    resp = await http.get(url, timeout=15.0)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

async def create_collection(qdrant: AsyncQdrantClient, recreate: bool = False):
    """
    Create the boots_catalog Qdrant collection with three named vector indexes.

    Schema per point:
      vectors = {
          "text_features": 3072-dim  (OpenAI, same value across all images for a SKU)
          "image":          768-dim  (CLIP, unique per image)
          "multimodal":     768-dim  (CLIP fused, unique per image)
      }
      payload = {
          sku, model_name, description, features[], category,
          price_usd, in_stock, color,      ← boot metadata
          image_index, image_url, total_images  ← per-image metadata
      }
    """
    existing = [c.name for c in (await qdrant.get_collections()).collections]

    if COLLECTION in existing:
        if recreate:
            await qdrant.delete_collection(COLLECTION)
            print(f"Deleted existing collection '{COLLECTION}'.")
        else:
            print(f"Collection '{COLLECTION}' already exists — skipping creation.")
            return

    await qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            TEXT_VEC:  VectorParams(size=TEXT_DIM,  distance=Distance.COSINE),
            IMAGE_VEC: VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
            MULTI_VEC: VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
        },
    )
    print(f"✓ Collection '{COLLECTION}' created with 3 named vector indexes:")
    print(f"    {TEXT_VEC!r}  → {TEXT_DIM}d  (OpenAI text-embedding-3-large)")
    print(f"    {IMAGE_VEC!r} → {IMAGE_DIM}d  (CLIP ViT-L/14 image)")
    print(f"    {MULTI_VEC!r} → {IMAGE_DIM}d  (CLIP ViT-L/14 fused image+text)")

    # Payload indexes allow fast pre-filtering before vector search
    payload_indexes = {
        "sku":         PayloadSchemaType.KEYWORD,
        "model_name":  PayloadSchemaType.KEYWORD,
        "category":    PayloadSchemaType.KEYWORD,
        "color":       PayloadSchemaType.KEYWORD,
        "in_stock":    PayloadSchemaType.BOOL,
        "price_usd":   PayloadSchemaType.FLOAT,
        "image_index": PayloadSchemaType.INTEGER,
    }
    for field, schema_type in payload_indexes.items():
        await qdrant.create_payload_index(COLLECTION, field, schema_type)
    print(f"✓ Payload indexes created: {list(payload_indexes.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_boot(boot: dict, qdrant: AsyncQdrantClient):
    """
    Ingest one boot record. Produces one Qdrant point per image URL.

    boot schema:
    {
        "sku":         str,           e.g. "8111"
        "model_name":  str,           e.g. "Iron Ranger"
        "description": str,
        "features":    list[str],     e.g. ["steel toe", "waterproof"]
        "category":    str,           e.g. "work" | "heritage" | "casual"
        "price_usd":   float,
        "in_stock":    bool,
        "color":       str,
        "image_urls":  list[str],     all product images for this SKU
    }

    Point ID is deterministic — uuid5(sku::imgN) — so re-ingestion is idempotent.
    """
    sku        = boot["sku"]
    image_urls = boot["image_urls"]

    # ── Step 1: Build text string and embed ONCE per SKU ──────────────────
    # This same vector is stored on ALL image points for this SKU.
    text_str = (
        f"{boot['model_name']}. "
        f"{boot['description']} "
        f"Features: {', '.join(boot['features'])}."
    )
    text_vec = await embed_text_openai(text_str)
    print(f"  [SKU {sku}] text embedded ({TEXT_DIM}d)")

    # ── Step 2: Embed each image, build one Qdrant point per image ────────
    points = []
    async with httpx.AsyncClient() as http:
        for idx, url in enumerate(image_urls):
            try:
                pil_img = await fetch_image_from_url(url, http)
            except Exception as e:
                print(f"  [WARN] SKU {sku} image {idx} failed to fetch: {e}")
                continue

            img_vec   = embed_image_clip(pil_img)
            multi_vec = fuse_image_text_clip(img_vec, text_str, alpha=0.6)

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sku}::img{idx}"))

            points.append(PointStruct(
                id=point_id,
                vectors={
                    TEXT_VEC:  text_vec,    # identical across all images of this SKU
                    IMAGE_VEC: img_vec,     # unique per image
                    MULTI_VEC: multi_vec,   # unique per image
                },
                payload={
                    # ── Boot metadata (same on every image point for this SKU) ──
                    "sku":          sku,
                    "model_name":   boot["model_name"],
                    "description":  boot["description"],
                    "features":     boot["features"],
                    "category":     boot["category"],
                    "price_usd":    boot["price_usd"],
                    "in_stock":     boot["in_stock"],
                    "color":        boot.get("color", ""),
                    # ── Per-image metadata ──
                    "image_index":  idx,
                    "image_url":    url,
                    "total_images": len(image_urls),
                },
            ))
            print(f"  [SKU {sku}] image {idx}/{len(image_urls)-1} embedded")

    if points:
        await qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"  ✓ Upserted {len(points)} points for SKU {sku}")
    else:
        print(f"  ✗ No valid images for SKU {sku} — skipping")


async def ingest_catalog(catalog: list[dict], recreate: bool = False):
    """Ingest full boot catalog. Processes boots concurrently in batches of 5."""
    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    await create_collection(qdrant, recreate=recreate)

    batch_size = 5
    for i in range(0, len(catalog), batch_size):
        batch = catalog[i : i + batch_size]
        print(f"\n── Batch {i // batch_size + 1} ({len(batch)} boots) ──")
        await asyncio.gather(*[ingest_boot(b, qdrant) for b in batch])

    await qdrant.close()
    print("\n✓ Catalog ingestion complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MAXSIM AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def maxsim_aggregate(raw_results: list, top_n: int) -> list[dict]:
    """
    Collapse image-level Qdrant results into unique boot SKUs.

    For each SKU, keep only the HIGHEST cosine similarity score across
    all its image points — that's the MaxSim operation.

    Example: SKU "8111" has 4 images. Three score 0.72, 0.68, 0.71, 0.85.
             After MaxSim, SKU "8111" gets score 0.85 and its best-matching
             image URL is recorded.

    Returns top_n boots sorted by score descending.
    """
    sku_best: dict[str, dict] = {}

    for point in raw_results:
        p     = point.payload
        sku   = p["sku"]
        score = point.score

        if sku not in sku_best or score > sku_best[sku]["score"]:
            sku_best[sku] = {
                "sku":               sku,
                "score":             round(score, 4),
                "model_name":        p["model_name"],
                "description":       p["description"],
                "features":          p["features"],
                "category":          p["category"],
                "price_usd":         p["price_usd"],
                "in_stock":          p["in_stock"],
                "color":             p["color"],
                "matched_image_url": p["image_url"],      # best-matching image angle
                "matched_img_index": p["image_index"],
                "total_images":      p["total_images"],
            }

    return sorted(sku_best.values(), key=lambda x: x["score"], reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
    """Convert a plain dict like {"in_stock": True, "category": "work"} to a Qdrant Filter."""
    if not filters:
        return None
    return Filter(must=[
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ])


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH — 3 MODES
# ─────────────────────────────────────────────────────────────────────────────

async def search_by_text(
    query: str,
    top_n: int = 10,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Search by feature text using the OpenAI text embedding index.

    Best for: "waterproof steel toe work boot", "insulated logger boot"
    Uses: TEXT_VEC index (3072-dim, text-embedding-3-large)

    Args:
        query:   Natural language feature description
        top_n:   Number of unique boots to return
        filters: Optional payload filters, e.g. {"in_stock": True, "category": "work"}

    Returns:
        List of boot dicts sorted by relevance score descending.
    """
    qdrant    = AsyncQdrantClient(url=QDRANT_URL)
    query_vec = await embed_text_openai(query)

    raw = await qdrant.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name=TEXT_VEC, vector=query_vec),
        query_filter=_build_filter(filters),
        limit=TOP_K_POINTS,
        with_payload=True,
    )

    results = maxsim_aggregate(raw, top_n)
    await qdrant.close()
    return results


async def search_by_image(
    query_image: Image.Image,
    top_n: int = 10,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Search by visual similarity using a query image.

    Best for: "find boots that look like this photo"
    Uses: IMAGE_VEC index (768-dim, CLIP ViT-L/14)

    Args:
        query_image: PIL Image (any size; preprocessing is handled internally)
        top_n:       Number of unique boots to return
        filters:     Optional payload filters

    Returns:
        List of boot dicts sorted by visual similarity score descending.
        'matched_image_url' shows which product angle was the closest match.
    """
    qdrant    = AsyncQdrantClient(url=QDRANT_URL)
    query_vec = embed_image_clip(query_image)

    raw = await qdrant.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name=IMAGE_VEC, vector=query_vec),
        query_filter=_build_filter(filters),
        limit=TOP_K_POINTS,
        with_payload=True,
    )

    results = maxsim_aggregate(raw, top_n)
    await qdrant.close()
    return results


async def search_by_image_and_text(
    query_image: Image.Image,
    query_text: str,
    top_n: int = 10,
    alpha: float = 0.6,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Multimodal search: image + text fused in CLIP's joint embedding space.

    Best for: photo of a boot + "slip resistant" or "in brown leather"
    Uses: MULTI_VEC index (768-dim, CLIP fused)

    How fusion works:
        CLIP encodes images and text into the SAME 768-dim vector space,
        so weighted addition is geometrically meaningful:
            fused = alpha * image_vec + (1 - alpha) * text_vec  (then L2-normalize)

    Args:
        query_image: PIL Image
        query_text:  Feature or style description to blend in
        top_n:       Number of unique boots to return
        alpha:       Image weight. 0.6 = 60% image / 40% text (default)
                     1.0 = image only | 0.0 = text only
        filters:     Optional payload filters

    Returns:
        List of boot dicts sorted by fused similarity score descending.
    """
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    img_vec   = np.array(embed_image_clip(query_image))
    txt_vec   = np.array(embed_text_clip(query_text))
    fused     = alpha * img_vec + (1 - alpha) * txt_vec
    fused     = (fused / np.linalg.norm(fused)).tolist()

    raw = await qdrant.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name=MULTI_VEC, vector=fused),
        query_filter=_build_filter(filters),
        limit=TOP_K_POINTS,
        with_payload=True,
    )

    results = maxsim_aggregate(raw, top_n)
    await qdrant.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Boot Visual Search API", version="1.0")


@app.post("/search/text", summary="Search by feature text")
async def api_search_text(
    query: str = Form(..., description="Feature description, e.g. 'waterproof steel toe'"),
    top_n: int = Form(10),
    category: Optional[str] = Form(None),
    in_stock: Optional[bool] = Form(None),
):
    filters = {k: v for k, v in {"category": category, "in_stock": in_stock}.items() if v is not None}
    results = await search_by_text(query, top_n=top_n, filters=filters or None)
    return JSONResponse({"query": query, "results": results, "count": len(results)})


@app.post("/search/image", summary="Search by boot image")
async def api_search_image(
    image: UploadFile = File(...),
    top_n: int = Form(10),
    category: Optional[str] = Form(None),
    in_stock: Optional[bool] = Form(None),
):
    pil     = Image.open(BytesIO(await image.read())).convert("RGB")
    filters = {k: v for k, v in {"category": category, "in_stock": in_stock}.items() if v is not None}
    results = await search_by_image(pil, top_n=top_n, filters=filters or None)
    return JSONResponse({"results": results, "count": len(results)})


@app.post("/search/multimodal", summary="Search by image + text")
async def api_search_multimodal(
    image: UploadFile = File(...),
    text: str = Form(..., description="Feature text to blend with the image"),
    alpha: float = Form(0.6, description="Image weight (0.0–1.0). 0.6 = 60% image, 40% text"),
    top_n: int = Form(10),
    in_stock: Optional[bool] = Form(None),
):
    pil     = Image.open(BytesIO(await image.read())).convert("RGB")
    filters = {"in_stock": in_stock} if in_stock is not None else None
    results = await search_by_image_and_text(pil, text, top_n=top_n, alpha=alpha, filters=filters)
    return JSONResponse({"text": text, "alpha": alpha, "results": results, "count": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CATALOG = [
    {
        "sku":         "8111",
        "model_name":  "Iron Ranger",
        "description": "Rugged cap-toe boot handcrafted for tough terrain and long days on your feet.",
        "features":    ["steel toe", "vibram sole", "full-grain leather", "goodyear welt"],
        "category":    "heritage",
        "price_usd":   389.0,
        "in_stock":    True,
        "color":       "amber harness",
        "image_urls":  [
            "https://example.com/8111_front.jpg",
            "https://example.com/8111_side.jpg",
            "https://example.com/8111_sole.jpg",
            "https://example.com/8111_back.jpg",
        ],
    },
    {
        "sku":         "2406",
        "model_name":  "Supersole 2.0",
        "description": "Lightweight composite-toe work boot built for all-day comfort with slip resistance.",
        "features":    ["composite toe", "waterproof", "slip resistant", "electrical hazard", "cushion insole"],
        "category":    "work",
        "price_usd":   249.0,
        "in_stock":    True,
        "color":       "brown",
        "image_urls":  [
            "https://example.com/2406_front.jpg",
            "https://example.com/2406_angle.jpg",
            "https://example.com/2406_side.jpg",
        ],
    },
    {
        "sku":         "3190",
        "model_name":  "Logger Pro",
        "description": "Chainsaw-rated logger boot with high ankle support for uneven ground.",
        "features":    ["steel toe", "waterproof", "insulated", "chainsaw protection", "lug sole"],
        "category":    "work",
        "price_usd":   459.0,
        "in_stock":    False,
        "color":       "black",
        "image_urls":  [
            "https://example.com/3190_front.jpg",
            "https://example.com/3190_side.jpg",
        ],
    },
    {
        "sku":         "6-inch-classic",
        "model_name":  "Classic 6-Inch",
        "description": "Everyday heritage boot with a clean silhouette. Transitions from work to weekend.",
        "features":    ["soft toe", "oil resistant", "full-grain leather", "cushion insole"],
        "category":    "casual",
        "price_usd":   199.0,
        "in_stock":    True,
        "color":       "tan",
        "image_urls":  [
            "https://example.com/classic6_front.jpg",
            "https://example.com/classic6_lifestyle.jpg",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE — run with: python boot_vector_search.py
# ─────────────────────────────────────────────────────────────────────────────

def print_results(label: str, results: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if not results:
        print("  No results.")
        return
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] SKU {r['sku']} — {r['model_name']}")
        print(f"       Category : {r['category']}  |  ${r['price_usd']}  |  In stock: {r['in_stock']}")
        print(f"       Features : {', '.join(r['features'])}")
        print(f"       Best img : {r['matched_image_url']}  (angle {r['matched_img_index']})")


async def main():
    # ── 1. Ingest catalog ─────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Ingest catalog")
    print("=" * 60)
    await ingest_catalog(SAMPLE_CATALOG, recreate=True)

    # ── 2. Search by text ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Search examples")
    print("=" * 60)

    text_results = await search_by_text(
        query="waterproof insulated work boot with steel toe",
        top_n=3,
        filters={"in_stock": True},
    )
    print_results(
        "TEXT SEARCH — 'waterproof insulated work boot with steel toe'  (in_stock=True)",
        text_results,
    )

    # ── 3. Search by image ────────────────────────────────────────────────
    # In real use: Image.open("uploaded_boot_photo.jpg")
    # Here we create a dummy gray image as a placeholder.
    dummy_query_image = Image.new("RGB", (224, 224), color=(120, 100, 80))

    image_results = await search_by_image(
        query_image=dummy_query_image,
        top_n=3,
    )
    print_results(
        "IMAGE SEARCH — query photo of a boot",
        image_results,
    )

    # ── 4. Search by image + text (multimodal) ────────────────────────────
    multi_results = await search_by_image_and_text(
        query_image=dummy_query_image,
        query_text="slip resistant electrical hazard composite toe",
        top_n=3,
        alpha=0.6,          # 60% image shape, 40% text features
        filters={"in_stock": True},
    )
    print_results(
        "MULTIMODAL SEARCH — photo + 'slip resistant electrical hazard composite toe'  alpha=0.6",
        multi_results,
    )

    # ── 5. Adjust alpha: text-heavy search ────────────────────────────────
    text_heavy_results = await search_by_image_and_text(
        query_image=dummy_query_image,
        query_text="heritage leather cap toe goodyear welt",
        top_n=3,
        alpha=0.3,          # 30% image, 70% text — let features dominate
    )
    print_results(
        "MULTIMODAL SEARCH — photo + 'heritage leather cap toe'  alpha=0.3 (text-heavy)",
        text_heavy_results,
    )


if __name__ == "__main__":
    asyncio.run(main())