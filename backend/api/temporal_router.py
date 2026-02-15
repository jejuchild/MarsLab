"""
Temporal Change Detection router.

Finds products covering the same area at different times,
enabling temporal comparison and change detection analysis.
"""

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/temporal", tags=["Temporal"])

# ODE instrument mapping: instrument_name -> (ihid, iid, product_types[])
# product_types is a list so we can query multiple ODE product types per instrument
_INSTRUMENT_MAP = {
    "HIRISE":        ("MRO", "HIRISE", ["RDRV11"]),
    "CTX":           ("MRO", "CTX",    ["EDR"]),
    "CRISM":         ("MRO", "CRISM",  ["MTRDR"]),
    "CRISM_TRR3":    ("MRO", "CRISM",  ["TRDR"]),
    "SHARAD":        ("MRO", "SHARAD", ["USRDRV2"]),
    "SHARAD_HIGHRES":("MRO", "SHARAD", ["RDR"]),
    "HIRISE_DTM":    ("MRO", "HIRISE", ["DTM"]),
}


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

    inst_upper = instrument.upper()
    if inst_upper not in _INSTRUMENT_MAP:
        return JSONResponse(
            {"error": f"Unsupported instrument: {instrument}. "
             f"Supported: {', '.join(sorted(_INSTRUMENT_MAP))}"},
            status_code=400,
        )

    ihid, iid, product_types = _INSTRUMENT_MAP[inst_upper]
    target = "mars"
    ode_lon = lon % 360  # ODE expects 0-360

    products = []

    async with httpx.AsyncClient(timeout=30) as client:
        for pt in product_types:
            url = (
                f"https://oderest.rsl.wustl.edu/live2/?target={target}"
                f"&query=product&results=p&output=JSON"
                f"&ihid={ihid}&iid={iid}&pt={pt}"
                f"&lat={lat}&lon={ode_lon}&loc=o&r={radius_km}"
                f"&limit=50"
            )

            try:
                resp = await client.get(url)
            except Exception as e:
                return JSONResponse(
                    {"pairs": [], "error": f"ODE query failed: {e}", "total_products": 0}
                )
            if resp.status_code != 200:
                continue

            data = resp.json()

            items = data.get("ODEResults", {}).get("Products", {})
            if items is None:
                continue
            item_list = items.get("Product", [])
            if isinstance(item_list, dict):
                item_list = [item_list]

            for item in item_list:
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
