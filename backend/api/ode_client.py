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
    Check if a string looks like a SHARAD product ID.

    SHARAD IDs typically look like:
    - S_00195401_THM (radargram thumbnail)
    - s_00195401 (partial ID)
    - s_001 (partial search query)
    """
    pattern = r"^S_\d+"
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
                f"results=p&"
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
            f"results=p&"
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

            products.append(ODEProduct(
                product_id=product_id,
                instrument=instrument,
                lat=lat,
                lon=lon,
                pdsid=p.get("pdsid"),
                ihid=p.get("IHID"),
                iid=p.get("IID"),
                pt=p.get("PT")
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
                        f"results=p&"
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
                    f"results=p&"
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
