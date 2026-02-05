"""
ODE (Orbital Data Explorer) client for querying Mars data.

Endpoints:
- ODE REST: https://oderest.rsl.wustl.edu/live2/
- ODE PRODUCTFILES: https://ode.rsl.wustl.edu/mars/productfiles.aspx

This module handles:
1. Product search (by ID or spatial coordinates)
2. File discovery via PRODUCTFILES
3. CRISM base_key parsing
4. Download URL resolution for CRISM and HiRISE bundles
"""

import re
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET


# =============================================================================
# Constants
# =============================================================================

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2"
ODE_PRODUCTFILES_BASE = "https://ode.rsl.wustl.edu/mars/productfiles.aspx"

# Instrument types we support
class Instrument(str, Enum):
    CRISM = "crism"
    HIRISE = "hirise"
    SHARAD = "sharad"
    SHARAD_HIGHRES = "sharad_highres"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ODEProduct:
    """Product returned from ODE search."""
    product_id: str
    instrument: Instrument
    lat: Optional[float] = None
    lon: Optional[float] = None
    # Additional metadata from ODE
    pdsid: Optional[str] = None
    ihid: Optional[str] = None  # Instrument host ID (MRO)
    iid: Optional[str] = None   # Instrument ID
    pt: Optional[str] = None    # Product type
    # Track start/stop coordinates (for SHARAD line products)
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    stop_lat: Optional[float] = None
    stop_lon: Optional[float] = None


@dataclass
class ODEFile:
    """File discovered from ODE PRODUCTFILES."""
    filename: str
    url: str
    file_type: str  # e.g., "Product", "Browse", "Derived"
    description: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass
class CRISMBundle:
    """Collection of files for a CRISM observation."""
    base_key: str
    # Core product files
    img_file: Optional[ODEFile] = None
    lbl_file: Optional[ODEFile] = None
    hdr_file: Optional[ODEFile] = None  # Optional ENVI header
    # Wavelength table
    tab_file: Optional[ODEFile] = None
    # Browse products
    browse_files: List[ODEFile] = field(default_factory=list)
    # Product type (MTR3 or TRR3)
    product_type: Optional[str] = None


@dataclass
class HiRISEBundle:
    """Collection of files for a HiRISE observation."""
    product_id: str
    # RED channel files
    jp2_file: Optional[ODEFile] = None
    lbl_file: Optional[ODEFile] = None


@dataclass
class SHARADBundle:
    """Collection of files for a SHARAD USRDRV2 observation (quicklook)."""
    product_id: str
    # Quicklook thumbnail
    thm_file: Optional[ODEFile] = None  # *_THM.JPG
    lbl_file: Optional[ODEFile] = None  # *.LBL
    # Geometry metadata
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    stop_lat: Optional[float] = None
    stop_lon: Optional[float] = None


@dataclass
class SHARADHighResBundle:
    """Collection of files for a SHARAD RDR (high-res) observation."""
    product_id: str
    # Binary data and label
    dat_file: Optional[ODEFile] = None  # *.DAT
    lbl_file: Optional[ODEFile] = None  # *.LBL
    # Orbit info
    orbit_number: Optional[int] = None


# =============================================================================
# CRISM Base Key Parsing
# =============================================================================

def parse_crism_base_key(product_id: str) -> str:
    """
    Extract base_key from a CRISM product ID.

    The base_key is the first two underscore-separated tokens, representing
    the observation-level identifier.

    Examples:
        frt0001fd76_07_if166j_mtr3 -> frt0001fd76_07
        frt00009312_07_if166l_trr3 -> frt00009312_07
        hrl00005a94_07_if182s_trr3 -> hrl00005a94_07

    Args:
        product_id: Full CRISM product ID

    Returns:
        Base key (observation-level identifier)
    """
    parts = product_id.lower().split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return product_id.lower()


def is_crism_product_id(product_id: str) -> bool:
    """
    Check if a string looks like a CRISM product ID.

    CRISM IDs typically start with:
    - frt (Full Resolution Targeted)
    - hrl (Half Resolution Long)
    - hrs (Half Resolution Short)
    - frs (Full Resolution Short)
    - arl (Along-track Resolution Low)
    """
    prefixes = ("frt", "hrl", "hrs", "frs", "arl", "atl")
    return product_id.lower().startswith(prefixes)


def is_hirise_product_id(product_id: str) -> bool:
    """
    Check if a string looks like a HiRISE product ID.

    HiRISE IDs typically look like:
    - ESP_045857_2350 (Extended Science Phase)
    - PSP_001234_5678 (Primary Science Phase)
    - ESP_045 (partial search query)
    """
    pattern = r"^(ESP|PSP|TRA)_\d+"
    return bool(re.match(pattern, product_id.upper()))


def is_sharad_product_id(product_id: str) -> bool:
    """
    Check if a string looks like a SHARAD USRDRV2 product ID (quicklook).

    SHARAD USRDRV2 IDs typically look like:
    - S_00195401_THM (radargram thumbnail)
    - s_00195401 (partial ID)
    - s_001 (partial search query)
    """
    pattern = r"^S_\d+"
    return bool(re.match(pattern, product_id.upper()))


def is_sharad_highres_product_id(product_id: str) -> bool:
    """
    Check if a string looks like a SHARAD High-Res RDR product ID.

    SHARAD RDR IDs typically look like:
    - R_5663601_001_SS19_700_A (Italian/ASI RDR format)
    - r_5663601 (partial ID)
    """
    pattern = r"^R_\d+"
    return bool(re.match(pattern, product_id.upper()))


# =============================================================================
# ODE REST API Queries
# =============================================================================

async def search_ode_products(
    query: str,
    instrument: Optional[Instrument] = None,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for products matching a query string (product ID or partial).

    Uses the ODE REST API to search for products. The query is matched against
    product IDs using a contains search.

    ODE REST Query Parameters:
    - target: mars
    - query: product identifier pattern
    - iid: Instrument ID (CRISM or HIRISE)
    - ihid: Instrument Host ID (MRO for Mars Reconnaissance Orbiter)
    - output: json
    - limit: max number of results

    Args:
        query: Search string (partial or full product ID)
        instrument: Optional filter by instrument type
        max_results: Maximum number of results to return
        session: Optional aiohttp session to reuse

    Returns:
        List of matching ODE products
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        products = []

        # Determine which instruments to search
        instruments_to_search = []
        if instrument:
            instruments_to_search = [instrument]
        else:
            # Auto-detect based on query pattern
            if is_crism_product_id(query):
                instruments_to_search = [Instrument.CRISM]
            elif is_hirise_product_id(query):
                instruments_to_search = [Instrument.HIRISE]
            else:
                # Search both
                instruments_to_search = [Instrument.CRISM, Instrument.HIRISE]

        for inst in instruments_to_search:
            inst_products = await _search_ode_by_instrument(
                query, inst, max_results, session
            )
            products.extend(inst_products)

        # Filter CRISM products to I/F reflectance data (_if) only
        if not instrument or instrument == Instrument.CRISM:
            products = [
                p for p in products
                if p.instrument != Instrument.CRISM or "_if" in p.product_id.lower()
            ]

        # Sort by relevance:
        # 1. Exact prefix match
        # 2. MTRDR products (map-projected, preferred)
        # 3. TRDR products
        # 4. Alphabetical within each group
        query_lower = query.lower()

        def sort_key(p: ODEProduct) -> tuple:
            pid_lower = p.product_id.lower()

            # Match priority: 0 = exact, 1 = prefix, 2 = contains
            if pid_lower == query_lower:
                match_score = 0
            elif pid_lower.startswith(query_lower):
                match_score = 1
            else:
                match_score = 2

            # Product type priority for CRISM: MTRDR (0) > TRDR (1) > other (2)
            if "_mtr" in pid_lower:
                type_score = 0
            elif "_trr" in pid_lower:
                type_score = 1
            else:
                type_score = 2

            return (match_score, type_score, pid_lower)

        products.sort(key=sort_key)
        return products[:max_results]

    finally:
        if close_session:
            await session.close()


async def _search_ode_by_instrument(
    query: str,
    instrument: Instrument,
    max_results: int,
    session: aiohttp.ClientSession
) -> List[ODEProduct]:
    """
    Search ODE REST API for a specific instrument.

    ODE REST endpoint for product search:
    https://oderest.rsl.wustl.edu/live2/?target=mars&ihid=mro&iid={instrument}&productid={query}*&output=json

    For CRISM, we search for processed I/F products (MTRDR, TRDR).
    For HiRISE, we search for RDR (reduced data) products.

    ODE REST API parameters:
    - results=p returns product list with pdsid (product ID)
    - pt=MTRDR/TRDR filters by product type for CRISM
    - productid with trailing wildcard for prefix matching
    """
    products = []

    if instrument == Instrument.CRISM:
        # Search for MTRDR (map-projected, preferred) and TRDR products
        # MTRDR = Map-projected Targeted Reduced Data Record
        # TRDR = Targeted Reduced Data Record (not map-projected)
        for pt in ["MTRDR", "TRDR"]:
            url = (
                f"{ODE_REST_BASE}?"
                f"target=mars&"
                f"ihid=mro&"
                f"iid=crism&"
                f"productid={query}*&"
                f"pt={pt}&"
                f"output=json&"
                f"results=pmf&"
                f"limit={max_results}"
            )
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        products.extend(_parse_ode_rest_response(data, instrument))
            except Exception as e:
                print(f"ODE REST search error for {instrument}/{pt}: {e}")

            # Stop if we have enough results
            if len(products) >= max_results:
                break

    else:  # HiRISE
        # Search for RDRV11 (RDR version 1.1) products
        url = (
            f"{ODE_REST_BASE}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=hirise&"
            f"productid={query}*&"
            f"output=json&"
            f"results=pmf&"
            f"limit={max_results}"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    products = _parse_ode_rest_response(data, instrument)
        except Exception as e:
            print(f"ODE REST search error for {instrument}: {e}")

    return products[:max_results]


def _parse_ode_rest_response(data: Dict[str, Any], instrument: Instrument) -> List[ODEProduct]:
    """
    Parse ODE REST JSON response into ODEProduct objects.

    ODE REST response structure:
    {
        "ODEResults": {
            "Status": "Success",
            "Count": N,
            "Products": {
                "Product": [
                    {
                        "pdsid": "...",
                        "IHID": "MRO",
                        "IID": "CRISM/HIRISE",
                        "PT": "product type",
                        "Footprint_C0_geometry": { ... },
                        ...
                    }
                ]
            }
        }
    }
    """
    products = []

    try:
        ode_results = data.get("ODEResults", {})
        if ode_results.get("Status") != "Success":
            return []

        products_data = ode_results.get("Products", {})
        product_list = products_data.get("Product", [])

        # Handle single product (ODE returns object instead of list for 1 result)
        if isinstance(product_list, dict):
            product_list = [product_list]

        for p in product_list:
            product_id = p.get("pdsid", p.get("PDSID", ""))

            # Extract coordinates from footprint if available
            lat, lon = None, None
            footprint = p.get("Footprint_C0_geometry") or p.get("Footprint")
            if footprint:
                lat = _extract_center_lat(footprint)
                lon = _extract_center_lon(footprint)

            # Also check for direct lat/lon fields
            if lat is None:
                lat = _safe_float(p.get("Center_latitude") or p.get("center_latitude"))
            if lon is None:
                lon = _safe_float(p.get("Center_longitude") or p.get("center_longitude"))

            # Extract start/stop coordinates for SHARAD track products
            start_lat, start_lon, stop_lat, stop_lon = None, None, None, None
            if instrument in (Instrument.SHARAD, Instrument.SHARAD_HIGHRES):
                start_lat = _safe_float(p.get("Start_latitude") or p.get("start_latitude"))
                start_lon = _safe_float(p.get("Start_longitude") or p.get("start_longitude"))
                stop_lat = _safe_float(p.get("Stop_latitude") or p.get("stop_latitude"))
                stop_lon = _safe_float(p.get("Stop_longitude") or p.get("stop_longitude"))

                # If start/stop not found, try to extract from LINESTRING footprint
                if start_lat is None and footprint:
                    coords = _extract_linestring_endpoints(footprint)
                    if coords:
                        start_lon, start_lat = coords[0]
                        stop_lon, stop_lat = coords[1]

            products.append(ODEProduct(
                product_id=product_id,
                instrument=instrument,
                lat=lat,
                lon=lon,
                pdsid=p.get("pdsid"),
                ihid=p.get("IHID"),
                iid=p.get("IID"),
                pt=p.get("PT"),
                start_lat=start_lat,
                start_lon=start_lon,
                stop_lat=stop_lat,
                stop_lon=stop_lon,
            ))

    except Exception as e:
        print(f"Error parsing ODE response: {e}")

    return products


async def search_ode_spatial(
    minlat: float,
    maxlat: float,
    westernlon: float,
    easternlon: float,
    instrument: Optional[Instrument] = None,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for products within a bounding box.

    ODE REST spatial query parameters:
    - minlat: Southern latitude boundary
    - maxlat: Northern latitude boundary
    - westernlon: Western longitude boundary
    - easternlon: Eastern longitude boundary

    Args:
        minlat: Southern latitude in degrees (-90 to 90)
        maxlat: Northern latitude in degrees (-90 to 90)
        westernlon: Western longitude in degrees (-180 to 360)
        easternlon: Eastern longitude in degrees (-180 to 360)
        instrument: Optional filter by instrument
        max_results: Maximum results to return
        session: Optional aiohttp session

    Returns:
        List of products within the bounding box
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        products = []

        instruments_to_search = [instrument] if instrument else [Instrument.CRISM, Instrument.HIRISE]

        for inst in instruments_to_search:
            # For CRISM, search MTRDR and TRDR product types
            if inst == Instrument.CRISM:
                for pt in ["MTRDR", "TRDR"]:
                    url = (
                        f"{ODE_REST_BASE}?"
                        f"target=mars&"
                        f"ihid=mro&"
                        f"iid={inst.value}&"
                        f"pt={pt}&"
                        f"minlat={minlat}&"
                        f"maxlat={maxlat}&"
                        f"westernlon={westernlon}&"
                        f"easternlon={easternlon}&"
                        f"output=json&"
                        f"results=pmf&"
                        f"limit={max_results}"
                    )

                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                inst_products = _parse_ode_rest_response(data, inst)
                                products.extend(inst_products)
                    except Exception as e:
                        print(f"ODE spatial search error for {inst}/{pt}: {e}")

                    if len(products) >= max_results:
                        break
            else:
                # HiRISE
                url = (
                    f"{ODE_REST_BASE}?"
                    f"target=mars&"
                    f"ihid=mro&"
                    f"iid={inst.value}&"
                    f"minlat={minlat}&"
                    f"maxlat={maxlat}&"
                    f"westernlon={westernlon}&"
                    f"easternlon={easternlon}&"
                    f"output=json&"
                    f"results=pmf&"
                    f"limit={max_results}"
                )

                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            inst_products = _parse_ode_rest_response(data, inst)
                            products.extend(inst_products)
                except Exception as e:
                    print(f"ODE spatial search error for {inst}: {e}")

        # Filter CRISM products to I/F reflectance data (_if) only
        products = [
            p for p in products
            if p.instrument != Instrument.CRISM or "_if" in p.product_id.lower()
        ]

        # Sort: MTRDR before TRDR
        products.sort(key=lambda p: (
            0 if "_mtr" in p.product_id.lower() else
            1 if "_trr" in p.product_id.lower() else 2,
            p.product_id.lower()
        ))

        return products[:max_results]

    finally:
        if close_session:
            await session.close()


# =============================================================================
# ODE PRODUCTFILES Discovery
# =============================================================================

async def discover_product_files(
    product_id: str,
    instrument: Instrument,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEFile]:
    """
    Discover all files associated with a product via ODE PRODUCTFILES.

    The PRODUCTFILES endpoint returns an XML document listing all files
    available for download for a given product.

    Endpoint:
    https://ode.rsl.wustl.edu/mars/productfiles.aspx?product_id={product_id}

    Response format (XML):
    <ODEFileList>
        <Product>
            <File>
                <URL>http://...</URL>
                <FileName>...</FileName>
                <KBytes>...</KBytes>
                <Type>Product/Browse/Derived</Type>
                <Description>...</Description>
            </File>
            ...
        </Product>
    </ODEFileList>

    Args:
        product_id: Product identifier to look up
        instrument: Instrument type
        session: Optional aiohttp session

    Returns:
        List of discovered files
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        url = f"{ODE_PRODUCTFILES_BASE}?productid={product_id}"

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return []

            xml_text = await resp.text()
            return _parse_productfiles_xml(xml_text)

    except Exception as e:
        print(f"ODE PRODUCTFILES error: {e}")
        return []
    finally:
        if close_session:
            await session.close()


def _parse_productfiles_xml(xml_text: str) -> List[ODEFile]:
    """Parse ODE PRODUCTFILES XML response."""
    files = []

    try:
        root = ET.fromstring(xml_text)

        # Find all File elements
        for file_elem in root.iter("File"):
            url = _get_elem_text(file_elem, "URL")
            filename = _get_elem_text(file_elem, "FileName")
            file_type = _get_elem_text(file_elem, "Type", "Unknown")
            description = _get_elem_text(file_elem, "Description")

            # Parse size (in KBytes)
            size_kb = _get_elem_text(file_elem, "KBytes")
            size_bytes = int(float(size_kb) * 1024) if size_kb else None

            if url and filename:
                files.append(ODEFile(
                    filename=filename,
                    url=url,
                    file_type=file_type,
                    description=description,
                    size_bytes=size_bytes
                ))

    except ET.ParseError as e:
        print(f"XML parse error: {e}")

    return files


def _get_elem_text(parent: ET.Element, tag: str, default: str = "") -> str:
    """Safely get text from an XML element."""
    elem = parent.find(tag)
    return elem.text if elem is not None and elem.text else default


# =============================================================================
# Bundle Resolution
# =============================================================================

async def _get_product_metadata(
    product_id: str,
    instrument: Instrument,
    session: aiohttp.ClientSession
) -> Optional[dict]:
    """Get product metadata including LabelURL from ODE REST API."""
    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&"
        f"ihid=mro&"
        f"iid={instrument.value}&"
        f"productid={product_id}&"
        f"output=json&"
        f"results=m&"
        f"limit=1"
    )

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            products = data.get("ODEResults", {}).get("Products", {}).get("Product")
            if isinstance(products, list):
                return products[0] if products else None
            return products
    except Exception as e:
        print(f"Error getting product metadata: {e}")
        return None


async def resolve_crism_bundle(
    base_key: str,
    session: Optional[aiohttp.ClientSession] = None
) -> CRISMBundle:
    """
    Resolve all files needed for a CRISM observation bundle.

    Given a base_key (e.g., frt0001fd76_07), discovers and organizes:
    1. Core product files (MTR3 or TRR3): *.img, *.lbl, *.hdr
    2. Wavelength table: *_wv*_mtr3.tab or *_wv*_trr3.tab
    3. Browse products: *_br*.png (HYD, ICE, IC2, VNA)

    Uses ODE REST API with results=m to get LabelURL, then constructs
    file URLs from the PDS archive path.

    Edge Cases:
    - Some products have MTR3 (map-projected), some have TRR3 (not map-projected)
    - HDR files may not exist for all products
    - Not all browse products exist for all observations

    Args:
        base_key: CRISM observation identifier (e.g., frt0001fd76_07)
        session: Optional aiohttp session

    Returns:
        CRISMBundle with resolved file URLs
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    bundle = CRISMBundle(base_key=base_key)

    try:
        # Search for I/F products matching this base_key
        products = await search_ode_products(base_key, Instrument.CRISM, 20, session)

        if not products:
            return bundle

        # Filter to find if* products (I/F reflectance data)
        if_products = [p for p in products if "_if" in p.product_id.lower()]

        # Sort to prefer MTR3 (map-projected) over TRR3
        if_products.sort(key=lambda p: (
            0 if "_mtr3" in p.product_id.lower() else
            1 if "_trr3" in p.product_id.lower() else 2
        ))

        if not if_products:
            return bundle

        # Get metadata for the best product to find LabelURL
        main_product = if_products[0]
        metadata = await _get_product_metadata(main_product.product_id, Instrument.CRISM, session)

        if not metadata:
            return bundle

        # Get LabelURL and construct file URLs
        label_url = metadata.get("LabelURL", "")
        if not label_url:
            return bundle

        # Determine product type
        if "_mtr3" in main_product.product_id.lower():
            bundle.product_type = "MTR3"
        elif "_trr3" in main_product.product_id.lower():
            bundle.product_type = "TRR3"

        # Construct base URL from LabelURL
        # LabelURL example: https://pds-geosciences.wustl.edu/.../frt0000a2c2_07_if166j_mtr3.xml
        base_url = label_url.rsplit("/", 1)[0] + "/"
        product_name = main_product.product_id.lower()

        # Core product files - construct URLs directly
        bundle.img_file = ODEFile(
            filename=f"{product_name}.img",
            url=f"{base_url}{product_name}.img",
            file_type="Product"
        )
        bundle.lbl_file = ODEFile(
            filename=f"{product_name}.lbl",
            url=f"{base_url}{product_name}.lbl",
            file_type="Product"
        )
        bundle.hdr_file = ODEFile(
            filename=f"{product_name}.hdr",
            url=f"{base_url}{product_name}.hdr",
            file_type="Product"
        )

        # Wavelength table - construct from product name pattern
        # e.g., frt0000a2c2_07_if166j_mtr3 -> frt0000a2c2_07_wv166j_mtr3
        wv_name = product_name.replace("_if", "_wv")
        bundle.tab_file = ODEFile(
            filename=f"{wv_name}.tab",
            url=f"{base_url}{wv_name}.tab",
            file_type="Product"
        )

        # Browse products - located in /browse/ directory instead of /mtrdr/
        # Pattern: /mtrdr/2008/... -> /browse/2008/...
        browse_base_url = base_url.replace("/mtrdr/", "/browse/").replace("/trdr/", "/browse/")

        # Common browse product types (quickview, hydration, ice, etc.)
        # Browse names: brtru (true color), brvna (volcanic thermal), brhyd (hydration), etc.
        browse_types = ["tru", "vna", "hyd", "ice", "ic2", "fal", "fem", "maf", "hcp", "lcp", "olv", "phs", "chl"]
        parts = product_name.split("_")
        if len(parts) >= 4:
            # Extract variant suffix (e.g., 'j' from 'if166j')
            variant_suffix = ""
            if parts[2] and parts[2][-1].isalpha():
                variant_suffix = parts[2][-1]

            for br_type in browse_types:
                # Pattern: frt0000a2c2_07_if166j_mtr3 -> frt0000a2c2_07_brtruj_mtr3
                br_name = f"{parts[0]}_{parts[1]}_br{br_type}{variant_suffix}_{parts[3]}"
                bundle.browse_files.append(ODEFile(
                    filename=f"{br_name}.png",
                    url=f"{browse_base_url}{br_name}.png",
                    file_type="Browse"
                ))

    except Exception as e:
        print(f"Error resolving CRISM bundle: {e}")
    finally:
        if close_session:
            await session.close()

    return bundle


async def resolve_hirise_bundle(
    product_id: str,
    session: Optional[aiohttp.ClientSession] = None
) -> HiRISEBundle:
    """
    Resolve files needed for a HiRISE observation.

    Given a product ID (e.g., ESP_045857_2350), discovers:
    1. RED JP2 image file
    2. RED label file

    HiRISE products have multiple versions:
    - EDR (Experiment Data Record): Raw data
    - RDR (Reduced Data Record): Calibrated, map-projected

    We download the RDR RED channel JP2, which is the primary
    grayscale science data product.

    Args:
        product_id: HiRISE product ID (e.g., ESP_045857_2350)
        session: Optional aiohttp session

    Returns:
        HiRISEBundle with resolved file URLs
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    bundle = HiRISEBundle(product_id=product_id)

    try:
        # Search for the RDR product
        # HiRISE RDR products have _RED suffix
        rdr_id = f"{product_id}_RED"

        files = await discover_product_files(rdr_id, Instrument.HIRISE, session)

        # If no files found, try without the _RED suffix
        if not files:
            files = await discover_product_files(product_id, Instrument.HIRISE, session)

        for f in files:
            fname_lower = f.filename.lower()

            # Look for RED JP2 file
            if "red" in fname_lower and fname_lower.endswith(".jp2"):
                bundle.jp2_file = f

            # Look for RED label
            elif "red" in fname_lower and fname_lower.endswith(".lbl"):
                bundle.lbl_file = f

    except Exception as e:
        print(f"Error resolving HiRISE bundle: {e}")
    finally:
        if close_session:
            await session.close()

    return bundle


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_center_lat(footprint: Any) -> Optional[float]:
    """Extract center latitude from a footprint geometry."""
    if isinstance(footprint, dict):
        # Handle various footprint formats
        if "center_latitude" in footprint:
            return _safe_float(footprint["center_latitude"])
        if "coordinates" in footprint:
            coords = footprint["coordinates"]
            if isinstance(coords, list) and len(coords) >= 2:
                # Assume [lon, lat] or [[lon, lat], ...]
                if isinstance(coords[0], list):
                    lats = [c[1] for c in coords if len(c) >= 2]
                    return sum(lats) / len(lats) if lats else None
                return _safe_float(coords[1])
    return None


def _extract_center_lon(footprint: Any) -> Optional[float]:
    """Extract center longitude from a footprint geometry."""
    if isinstance(footprint, dict):
        if "center_longitude" in footprint:
            return _safe_float(footprint["center_longitude"])
        if "coordinates" in footprint:
            coords = footprint["coordinates"]
            if isinstance(coords, list) and len(coords) >= 2:
                if isinstance(coords[0], list):
                    lons = [c[0] for c in coords if len(c) >= 2]
                    return sum(lons) / len(lons) if lons else None
                return _safe_float(coords[0])
    return None


def _extract_linestring_endpoints(footprint: Any) -> Optional[List[List[float]]]:
    """
    Extract start and end points from a footprint geometry (for SHARAD tracks).

    Args:
        footprint: Footprint geometry (dict with coordinates or WKT string)

    Returns:
        List of [[start_lon, start_lat], [stop_lon, stop_lat]] or None
    """
    coords = None

    # Handle WKT string
    if isinstance(footprint, str) and "LINESTRING" in footprint.upper():
        coords = _parse_wkt_linestring(footprint)

    # Handle dict with coordinates
    elif isinstance(footprint, dict) and "coordinates" in footprint:
        c = footprint["coordinates"]
        if isinstance(c, list) and len(c) >= 2 and isinstance(c[0], list):
            coords = c

    if coords and len(coords) >= 2:
        return [coords[0], coords[-1]]  # [start, end]

    return None


def _parse_wkt_linestring(wkt: str) -> List[List[float]]:
    """
    Parse WKT LINESTRING into list of [lon, lat] coordinates.

    Args:
        wkt: WKT string like "LINESTRING (-165.4 51.2, -165.5 50.9, ...)"

    Returns:
        List of [lon, lat] pairs
    """
    import re

    # Extract coordinates from LINESTRING (lon lat, lon lat, ...)
    match = re.search(r'LINESTRING\s*\((.*)\)', wkt, re.IGNORECASE)
    if not match:
        return []

    coords_str = match.group(1)
    coordinates = []

    for pair in coords_str.split(','):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coordinates.append([lon, lat])
            except ValueError:
                continue

    return coordinates


async def get_sharad_highres_footprint(
    product_id: str,
    session: Optional[aiohttp.ClientSession] = None
) -> List[List[float]]:
    """
    Fetch footprint coordinates for a SHARAD High-Res product from ODE.

    Args:
        product_id: SHARAD RDR product ID (e.g., R_3898101_001_SS19_700_A)
        session: Optional aiohttp session

    Returns:
        List of [lon, lat] coordinates forming the track LineString
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    coordinates = []

    try:
        url = (
            f"{ODE_REST_BASE}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"productid={product_id}*&"
            f"pt=RDR&"
            f"output=json&"
            f"results=pmf&"
            f"limit=1"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
                if isinstance(products, dict):
                    products = [products]

                if products:
                    p = products[0]
                    # Try Footprint_C0_geometry first (WKT LINESTRING)
                    footprint_wkt = p.get("Footprint_C0_geometry", "")
                    if footprint_wkt and "LINESTRING" in footprint_wkt.upper():
                        coordinates = _parse_wkt_linestring(footprint_wkt)

                    # Fallback to start/stop coordinates if WKT parsing fails
                    if not coordinates:
                        start_lat = _safe_float(p.get("Start_latitude"))
                        start_lon = _safe_float(p.get("Start_longitude"))
                        stop_lat = _safe_float(p.get("Stop_latitude"))
                        stop_lon = _safe_float(p.get("Stop_longitude"))

                        if all(v is not None for v in [start_lat, start_lon, stop_lat, stop_lon]):
                            coordinates = [[start_lon, start_lat], [stop_lon, stop_lat]]

    except Exception as e:
        print(f"Error fetching SHARAD High-Res footprint: {e}")
    finally:
        if close_session:
            await session.close()

    return coordinates


# =============================================================================
# SHARAD Bundle Resolution
# =============================================================================

# ODE endpoints for SHARAD products
ODE_SHARAD_SEARCH = "https://oderest.rsl.wustl.edu/live2"
# PDS Geosciences Node SHARAD data archive
PDS_SHARAD_BASE = "https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/browse/thm"


async def search_sharad_spatial(
    minlat: float,
    maxlat: float,
    westernlon: float,
    easternlon: float,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for SHARAD USRDRV2 products within a bounding box.

    Args:
        minlat: Southern latitude boundary
        maxlat: Northern latitude boundary
        westernlon: Western longitude boundary
        easternlon: Eastern longitude boundary
        max_results: Maximum results to return
        session: Optional aiohttp session

    Returns:
        List of SHARAD products within the bounding box
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    products = []

    try:
        url = (
            f"{ODE_SHARAD_SEARCH}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"pt=USRDRV2&"
            f"minlat={minlat}&"
            f"maxlat={maxlat}&"
            f"westernlon={westernlon}&"
            f"easternlon={easternlon}&"
            f"output=json&"
            f"results=pmf&"
            f"limit={max_results}"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = _parse_ode_rest_response(data, Instrument.SHARAD)

    except Exception as e:
        print(f"SHARAD spatial search error: {e}")
    finally:
        if close_session:
            await session.close()

    return products[:max_results]


async def search_sharad_highres_spatial(
    minlat: float,
    maxlat: float,
    westernlon: float,
    easternlon: float,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for SHARAD High-Res RDR products within a bounding box.

    Args:
        minlat: Southern latitude boundary
        maxlat: Northern latitude boundary
        westernlon: Western longitude boundary
        easternlon: Eastern longitude boundary
        max_results: Maximum results to return
        session: Optional aiohttp session

    Returns:
        List of SHARAD High-Res products within the bounding box
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    products = []

    try:
        url = (
            f"{ODE_SHARAD_SEARCH}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"pt=RDR&"
            f"minlat={minlat}&"
            f"maxlat={maxlat}&"
            f"westernlon={westernlon}&"
            f"easternlon={easternlon}&"
            f"output=json&"
            f"results=pmf&"
            f"limit={max_results}"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = _parse_ode_rest_response(data, Instrument.SHARAD_HIGHRES)

    except Exception as e:
        print(f"SHARAD High-Res spatial search error: {e}")
    finally:
        if close_session:
            await session.close()

    return products[:max_results]


async def search_sharad_products(
    query: str,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for SHARAD USRDRV2 products.

    SHARAD USRDRV2 are quicklook radargram thumbnails (THM.JPG).

    Args:
        query: Search string (partial or full product ID)
        max_results: Maximum number of results
        session: Optional aiohttp session

    Returns:
        List of matching SHARAD products
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    products = []

    try:
        # Search for USRDR (US reduced data record) products
        url = (
            f"{ODE_SHARAD_SEARCH}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"productid={query}*&"
            f"pt=USRDRV2&"  # US Reduced Data Record V2 (quicklook)
            f"output=json&"
            f"results=pmf&"
            f"limit={max_results}"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = _parse_ode_rest_response(data, Instrument.SHARAD)

    except Exception as e:
        print(f"SHARAD search error: {e}")
    finally:
        if close_session:
            await session.close()

    return products[:max_results]


async def search_sharad_highres_products(
    query: str,
    max_results: int = 10,
    session: Optional[aiohttp.ClientSession] = None
) -> List[ODEProduct]:
    """
    Search ODE for SHARAD High-Res RDR products.

    SHARAD RDR are Italian/ASI pipeline products with full radargram data.

    Args:
        query: Search string (partial or full product ID)
        max_results: Maximum number of results
        session: Optional aiohttp session

    Returns:
        List of matching SHARAD High-Res products
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    products = []

    try:
        # Search for RDR (Reduced Data Record) products from Italian pipeline
        url = (
            f"{ODE_SHARAD_SEARCH}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"productid={query}*&"
            f"pt=RDR&"  # Italian RDR products
            f"output=json&"
            f"results=pmf&"
            f"limit={max_results}"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = _parse_ode_rest_response(data, Instrument.SHARAD_HIGHRES)

    except Exception as e:
        print(f"SHARAD High-Res search error: {e}")
    finally:
        if close_session:
            await session.close()

    return products[:max_results]


async def resolve_sharad_bundle(
    product_id: str,
    session: Optional[aiohttp.ClientSession] = None
) -> SHARADBundle:
    """
    Resolve files needed for a SHARAD USRDRV2 observation.

    Given a product ID (e.g., S_00195401), discovers:
    1. THM.JPG quicklook thumbnail
    2. LBL label file

    USRDRV2 products are stored in the PDS archive at:
    https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/

    Args:
        product_id: SHARAD product ID (e.g., S_00195401_THM or S_00195401)
        session: Optional aiohttp session

    Returns:
        SHARADBundle with resolved file URLs
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    # Normalize product ID: extract base (S_XXXXXXXX)
    pid_upper = product_id.upper()
    if "_THM" in pid_upper:
        base_id = pid_upper.replace("_THM", "")
    else:
        base_id = pid_upper

    bundle = SHARADBundle(product_id=base_id)

    try:
        # Use ODE PRODUCTFILES to discover URLs
        files = await discover_product_files(product_id, Instrument.SHARAD, session)

        for f in files:
            fname_lower = f.filename.lower()

            if fname_lower.endswith("_thm.jpg") or fname_lower.endswith("thm.jpg"):
                bundle.thm_file = f
            elif fname_lower.endswith(".lbl"):
                bundle.lbl_file = f

        # If no files found via PRODUCTFILES, construct URLs directly
        if not bundle.thm_file:
            # SHARAD USRDRV2 path pattern:
            # /mro/mro-m-sharad-5-radargram-v2/mrosh_2101/browse/thm/s_00195401_thm.jpg
            orbit = base_id.split("_")[1] if "_" in base_id else ""
            thm_filename = f"{base_id.lower()}_thm.jpg"
            lbl_filename = f"{base_id.lower()}_thm.lbl"

            # Construct URL based on orbit number grouping
            if orbit:
                orbit_num = int(orbit)
                # Files are grouped by orbit ranges
                orbit_dir = f"{(orbit_num // 1000) * 1000:05d}_{((orbit_num // 1000) + 1) * 1000 - 1:05d}"
                base_url = f"https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/browse/thm/{orbit_dir}"

                bundle.thm_file = ODEFile(
                    filename=thm_filename,
                    url=f"{base_url}/{thm_filename}",
                    file_type="Browse"
                )
                bundle.lbl_file = ODEFile(
                    filename=lbl_filename,
                    url=f"{base_url}/{lbl_filename}",
                    file_type="Label"
                )

    except Exception as e:
        print(f"Error resolving SHARAD bundle: {e}")
    finally:
        if close_session:
            await session.close()

    return bundle


async def resolve_sharad_highres_bundle(
    product_id: str,
    session: Optional[aiohttp.ClientSession] = None
) -> SHARADHighResBundle:
    """
    Resolve files needed for a SHARAD High-Res RDR observation.

    Given a product ID (e.g., R_5663601_001_SS19_700_A), discovers:
    1. DAT binary data file
    2. LBL label file

    Uses ODE REST API with results=m to get the LabelURL and construct file paths.

    Args:
        product_id: SHARAD RDR product ID (e.g., R_5663601_001_SS19_700_A)
        session: Optional aiohttp session

    Returns:
        SHARADHighResBundle with resolved file URLs
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    bundle = SHARADHighResBundle(product_id=product_id.upper())

    try:
        # Extract orbit number from product ID
        # Format: R_OOOOOOO_SSS_TTNN_RRR_V (orbit_sequence_table_range_version)
        parts = product_id.upper().split("_")
        if len(parts) >= 2:
            try:
                bundle.orbit_number = int(parts[1])
            except ValueError:
                pass

        # Use ODE REST API to get LabelURL (more reliable than PRODUCTFILES)
        url = (
            f"{ODE_REST_BASE}?"
            f"target=mars&"
            f"ihid=mro&"
            f"iid=sharad&"
            f"productid={product_id}*&"
            f"pt=RDR&"
            f"output=json&"
            f"results=m&"
            f"limit=1"
        )

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
                if isinstance(products, dict):
                    products = [products]

                if products:
                    p = products[0]
                    label_url = p.get("LabelURL", "")

                    if label_url:
                        # Construct file URLs from LabelURL
                        # LabelURL: .../r_0401101_001_ss19_700_a.lbl
                        base_url = label_url.rsplit("/", 1)[0]
                        pid_lower = product_id.lower()

                        bundle.lbl_file = ODEFile(
                            filename=f"{pid_lower}.lbl",
                            url=label_url,
                            file_type="Label"
                        )
                        bundle.dat_file = ODEFile(
                            filename=f"{pid_lower}.dat",
                            url=f"{base_url}/{pid_lower}.dat",
                            file_type="Product"
                        )

    except Exception as e:
        print(f"Error resolving SHARAD High-Res bundle: {e}")
    finally:
        if close_session:
            await session.close()

    return bundle
