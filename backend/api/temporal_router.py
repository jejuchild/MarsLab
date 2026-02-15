"""
Temporal Change Detection router.

Finds products covering the same area at different times,
enabling temporal comparison and change detection analysis.
"""

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/temporal", tags=["Temporal"])


@router.post("/find_pairs")
async def find_temporal_pairs(
    lat: float = Body(...),
    lon: float = Body(...),
    radius_km: float = Body(50),
    instrument: str = Body("HIRISE"),
):
    """
    Find products covering the same area at different times.
    Returns pairs of products with their observation dates.
    """
    import httpx

    # Validate inputs
    if not (-90 <= lat <= 90):
        return JSONResponse(
            {"error": "Latitude must be between -90 and 90"},
            status_code=400,
        )
    if not (0 < radius_km <= 500):
        return JSONResponse(
            {"error": "radius_km must be between 0 and 500"},
            status_code=400,
        )

    target = "mars"
    if instrument == "HIRISE":
        ihid = "MRO"
        iid = "HIRISE"
        pt = "RDRV11"
    elif instrument == "CTX":
        ihid = "MRO"
        iid = "CTX"
        pt = "EDR"
    else:
        return JSONResponse(
            {"error": f"Unsupported instrument: {instrument}"},
            status_code=400,
        )

    url = (
        f"https://oderest.rsl.wustl.edu/live2/?target={target}"
        f"&query=product&results=p&output=JSON"
        f"&ihid={ihid}&iid={iid}&pt={pt}"
        f"&lat={lat}&lon={lon % 360}&loc=o&r={radius_km}"
        f"&limit=50"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(url)
        except Exception as e:
            return JSONResponse(
                {"pairs": [], "error": f"ODE query failed: {e}", "total_products": 0}
            )
        if resp.status_code != 200:
            return JSONResponse(
                {"pairs": [], "error": "ODE query failed", "total_products": 0}
            )

        data = resp.json()

    products = []
    items = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
    if isinstance(items, dict):
        items = [items]

    for item in items:
        pid = item.get("pdsid", item.get("product_id", ""))
        ut_time = item.get("UTC_start_time", "")

        products.append({
            "product_id": pid,
            "date": ut_time[:10] if ut_time else "Unknown",
            "full_date": ut_time,
        })

    # Sort by date
    products.sort(key=lambda p: p["date"])

    # Find pairs (products at different times covering same area)
    pairs = []
    for i in range(len(products)):
        for j in range(i + 1, min(i + 5, len(products))):
            if products[i]["date"] != products[j]["date"]:
                pairs.append({
                    "product_a": products[i],
                    "product_b": products[j],
                    "time_gap_info": f"{products[i]['date']} -> {products[j]['date']}",
                })

    return JSONResponse(content={
        "pairs": pairs[:10],
        "total_products": len(products),
    })
