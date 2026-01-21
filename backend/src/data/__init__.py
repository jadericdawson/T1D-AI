"""Data module for T1D-AI - contains static data like GI database."""

from data.glycemic_index_db import (
    GLYCEMIC_INDEX_DATABASE,
    lookup_gi,
    get_all_foods,
    get_foods_by_category,
    get_categories,
)

__all__ = [
    "GLYCEMIC_INDEX_DATABASE",
    "lookup_gi",
    "get_all_foods",
    "get_foods_by_category",
    "get_categories",
]
