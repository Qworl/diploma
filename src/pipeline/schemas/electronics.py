"""Electronics domain schema and few-shot examples for LLM prompts."""

ELECTRONICS_SCHEMA = {
    "brand": {
        "type": "enum",
        "values": ["Apple", "Samsung", "Xiaomi", "Huawei", "Oppo",
                   "Vivo", "OnePlus", "Google", "Sony", "Other"],
        "description": "Phone manufacturer",
    },
    "os": {
        "type": "enum",
        "values": ["iOS", "Android", "HarmonyOS", "Other"],
        "description": "Mobile operating system",
    },
    "form_factor": {
        "type": "enum",
        "values": ["bar", "foldable", "flip", "rugged", "other"],
        "description": "Phone form factor",
    },
    "screen_size_class": {
        "type": "enum",
        "values": ["small", "medium", "large", "phablet"],
        "nullable": True,
        "description": "Screen size: small <5in, medium 5-6in, large 6-6.7in, phablet >6.7in",
    },
    "ram_class": {
        "type": "enum",
        "values": ["2GB", "4GB", "6GB", "8GB", "12GB+"],
        "nullable": True,
        "description": "RAM tier",
    },
    "storage_class": {
        "type": "enum",
        "values": ["32GB", "64GB", "128GB", "256GB", "512GB+"],
        "nullable": True,
        "description": "Storage tier",
    },
    "price_tier": {
        "type": "enum",
        "values": ["budget", "midrange", "premium", "flagship"],
        "nullable": True,
        "description": "Price tier: budget <$200, midrange $200-500, premium $500-1000, flagship $1000+",
    },
    "release_year_class": {
        "type": "enum",
        "values": ["pre-2020", "2020-2022", "2023-2024", "2025+"],
        "nullable": True,
        "description": "Release year bucket",
    },
}

ELECTRONICS_EXAMPLES = [
    (
        {
            "product_name": "Apple iPhone 15 Pro 256GB Titanium",
            "brands": "Apple",
        },
        {
            "brand": "Apple",
            "os": "iOS",
            "form_factor": "bar",
            "screen_size_class": "large",
            "ram_class": "8GB",
            "storage_class": "256GB",
            "price_tier": "flagship",
            "release_year_class": "2023-2024",
        },
    ),
    (
        {
            "product_name": "Samsung Galaxy Z Fold5 12/512GB",
            "brands": "Samsung",
        },
        {
            "brand": "Samsung",
            "os": "Android",
            "form_factor": "foldable",
            "screen_size_class": "phablet",
            "ram_class": "12GB+",
            "storage_class": "512GB+",
            "price_tier": "flagship",
            "release_year_class": "2023-2024",
        },
    ),
    (
        {
            "product_name": "Xiaomi Redmi 9A 2/32GB",
            "brands": "Xiaomi",
        },
        {
            "brand": "Xiaomi",
            "os": "Android",
            "form_factor": "bar",
            "screen_size_class": "large",
            "ram_class": "2GB",
            "storage_class": "32GB",
            "price_tier": "budget",
            "release_year_class": "2020-2022",
        },
    ),
]
