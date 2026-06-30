"""
postprocess.py — Post-processing and normalization of validated LLM output.

After Pydantic validation, data may still contain minor inconsistencies:
    - Brand names capitalised differently ("apple" vs "Apple" vs "APPLE")
    - Category strings not matching canonical values
    - Currency strings still containing symbols
    - Availability strings with typos/abbreviations

This module applies deterministic, rule-based normalization so the final
CSV is fully consistent and ready for downstream analytics.
"""

from __future__ import annotations

import re

from src.logger import get_logger
from src.schemas import NormalizedProduct

log = get_logger(__name__)

# ── Brand normalisation map ───────────────────────────────────────────────────
# Keys are lowercase aliases → canonical display value
BRAND_MAP: dict[str, str] = {
    # Mobile / Tech
    "apple": "Apple",
    "iphone": "Apple",
    "samsung": "Samsung",
    "oneplus": "OnePlus",
    "one plus": "OnePlus",
    "1+": "OnePlus",
    "xiaomi": "Xiaomi",
    "mi": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "realme": "Realme",
    "oppo": "Oppo",
    "vivo": "Vivo",
    "nokia": "Nokia",
    "motorola": "Motorola",
    "moto": "Motorola",
    "google": "Google",
    "pixel": "Google",
    "nothing": "Nothing",
    "iqoo": "iQOO",
    # Laptops
    "dell": "Dell",
    "hp": "HP",
    "hewlett-packard": "HP",
    "lenovo": "Lenovo",
    "asus": "Asus",
    "acer": "Acer",
    "msi": "MSI",
    "lg": "LG",
    "macbook": "Apple",
    # Audio
    "sony": "Sony",
    "bose": "Bose",
    "jbl": "JBL",
    "sennheiser": "Sennheiser",
    "boat": "boAt",
    "noise": "Noise",
    "skullcandy": "Skullcandy",
    "jabra": "Jabra",
    "anker": "Anker",
    # TV
    "tcl": "TCL",
    "hisense": "Hisense",
    "vu": "Vu",
    "mi tv": "Xiaomi",
    # Shoes / Fashion
    "nike": "Nike",
    "adidas": "Adidas",
    "puma": "Puma",
    "reebok": "Reebok",
    "asics": "ASICS",
    "new balance": "New Balance",
    "skechers": "Skechers",
    "bata": "Bata",
    "woodland": "Woodland",
    "campus": "Campus",
    "levi's": "Levi's",
    "levis": "Levi's",
    "h&m": "H&M",
    "zara": "Zara",
    "allen solly": "Allen Solly",
    # Kitchen / Appliances
    "philips": "Philips",
    "havells": "Havells",
    "bajaj": "Bajaj",
    "prestige": "Prestige",
    "hawkins": "Hawkins",
    "pigeon": "Pigeon",
    "lifelong": "Lifelong",
    "inalsa": "Inalsa",
    "morphy richards": "Morphy Richards",
    "wonderchef": "Wonderchef",
    "kent": "Kent",
    # Gaming
    "razer": "Razer",
    "corsair": "Corsair",
    "steelseries": "SteelSeries",
    "hyperx": "HyperX",
    "logitech": "Logitech",
    "playstation": "Sony",
    "ps5": "Sony",
    "xbox": "Microsoft",
    "nintendo": "Nintendo",
    # Books
    "penguin": "Penguin",
    "harper collins": "HarperCollins",
    "harpercollins": "HarperCollins",
    "oxford": "Oxford University Press",
    "scholastic": "Scholastic",
    # Furniture
    "ikea": "IKEA",
    "urban ladder": "Urban Ladder",
    "pepperfry": "Pepperfry",
    "godrej": "Godrej",
    "nilkamal": "Nilkamal",
    "durian": "Durian",
}

# ── Category normalisation map ─────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str] = {
    "mobile": "Mobile",
    "smartphone": "Mobile",
    "phone": "Mobile",
    "cellphone": "Mobile",
    "iphone": "Mobile",
    "android": "Mobile",
    "laptop": "Laptop",
    "notebook": "Laptop",
    "macbook": "Laptop",
    "chromebook": "Laptop",
    "ultrabook": "Laptop",
    "headphone": "Headphones",
    "headphones": "Headphones",
    "earphones": "Headphones",
    "earbuds": "Headphones",
    "tws": "Headphones",
    "headset": "Headphones",
    "neckband": "Headphones",
    "speaker": "Headphones",
    "tv": "TV",
    "television": "TV",
    "smart tv": "TV",
    "oled": "TV",
    "qled": "TV",
    "shoes": "Shoes",
    "sneakers": "Shoes",
    "footwear": "Shoes",
    "sandals": "Shoes",
    "boots": "Shoes",
    "running shoes": "Shoes",
    "fashion": "Fashion",
    "clothing": "Fashion",
    "apparel": "Fashion",
    "shirt": "Fashion",
    "jeans": "Fashion",
    "dress": "Fashion",
    "t-shirt": "Fashion",
    "jacket": "Fashion",
    "kitchen appliance": "Kitchen Appliance",
    "kitchen": "Kitchen Appliance",
    "appliance": "Kitchen Appliance",
    "cookware": "Kitchen Appliance",
    "mixer": "Kitchen Appliance",
    "blender": "Kitchen Appliance",
    "air fryer": "Kitchen Appliance",
    "pressure cooker": "Kitchen Appliance",
    "gaming": "Gaming",
    "game": "Gaming",
    "console": "Gaming",
    "controller": "Gaming",
    "gaming mouse": "Gaming",
    "gaming keyboard": "Gaming",
    "book": "Book",
    "novel": "Book",
    "textbook": "Book",
    "ebook": "Book",
    "fiction": "Book",
    "non-fiction": "Book",
    "furniture": "Furniture",
    "sofa": "Furniture",
    "chair": "Furniture",
    "table": "Furniture",
    "bed": "Furniture",
    "wardrobe": "Furniture",
    "shelf": "Furniture",
    "unknown": "Unknown",
}

# ── Availability normalisation ────────────────────────────────────────────────
AVAILABILITY_MAP: dict[str, str] = {
    "in stock": "In Stock",
    "instock": "In Stock",
    "available": "In Stock",
    "in-stock": "In Stock",
    "yes": "In Stock",
    "out of stock": "Out of Stock",
    "outofstock": "Out of Stock",
    "out-of-stock": "Out of Stock",
    "sold out": "Out of Stock",
    "no": "Out of Stock",
    "unavailable": "Out of Stock",
    "coming soon": "Coming Soon",
    "upcoming": "Coming Soon",
    "preorder": "Coming Soon",
    "pre-order": "Coming Soon",
    "notify me": "Coming Soon",
    "limited stock": "Limited Stock",
    "limited": "Limited Stock",
    "hurry": "Limited Stock",
    "only 1 left": "Limited Stock",
    "only 2 left": "Limited Stock",
    "only 3 left": "Limited Stock",
    "few left": "Limited Stock",
    "unknown": "Unknown",
}


# ── Normalisation functions ────────────────────────────────────────────────────

def normalize_brand(brand: str) -> str:
    """Map brand string to its canonical form using BRAND_MAP."""
    return BRAND_MAP.get(brand.lower().strip(), brand.strip().title())


def normalize_category(category: str) -> str:
    """Map category string to its canonical form using CATEGORY_MAP."""
    key = category.lower().strip()
    # Direct lookup
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    # Substring lookup (e.g., "wireless headphones" → "Headphones")
    for alias, canonical in CATEGORY_MAP.items():
        if alias in key:
            return canonical
    return category.strip().title()


def normalize_availability(availability: str) -> str:
    """Map availability string to its canonical form using AVAILABILITY_MAP."""
    key = availability.lower().strip()
    if key in AVAILABILITY_MAP:
        return AVAILABILITY_MAP[key]
    # Pattern: "only N left" → "Limited Stock"
    if re.match(r"only \d+ left", key):
        return "Limited Stock"
    return availability.strip()


def normalize_offer(offer: str | None) -> str | None:
    """Lightly clean offer text: strip, fix case, remove double spaces."""
    if not offer:
        return None
    cleaned = re.sub(r"\s+", " ", offer.strip())
    return cleaned if cleaned else None


def normalize_seller(seller: str | None) -> str | None:
    """Title-case seller name if present."""
    if not seller:
        return None
    return seller.strip().title()


def postprocess_product(product: NormalizedProduct) -> NormalizedProduct:
    """
    Apply all normalization rules to a single validated product record.

    Returns a new :class:`NormalizedProduct` with normalised fields
    (uses ``model_copy`` to avoid mutating the original).
    """
    updates = {
        "brand": normalize_brand(product.brand),
        "category": normalize_category(product.category),
        "availability": normalize_availability(product.availability),
        "offer": normalize_offer(product.offer),
        "seller": normalize_seller(product.seller),
        "currency": product.currency.upper() if product.currency else "INR",
    }
    return product.model_copy(update=updates)


def postprocess_batch(products: list[NormalizedProduct]) -> list[NormalizedProduct]:
    """
    Apply post-processing to an entire batch.

    Args:
        products: List of validated :class:`NormalizedProduct` objects.

    Returns:
        New list with all normalization rules applied.
    """
    processed = [postprocess_product(p) for p in products]
    log.info(f"Post-processing complete — {len(processed)} records normalized")
    return processed
