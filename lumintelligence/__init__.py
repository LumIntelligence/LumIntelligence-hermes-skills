"""LumIntelligence — four-tier knowledge-network library (Engine).

Ingests source units, discovers entities (NoD), tags them into concepts, consolidates concepts
into centroids, and builds a graph + vector store with deterministic source lineage. The
``LumIntelligence`` facade is the entry point.
"""

__version__ = "0.1.2"

from lumintelligence.config import Config, OpenAIConfig
from lumintelligence.facade import LumIntelligence
from lumintelligence.graph import GraphStore, SqliteGraphStore
from lumintelligence.prompts import load_prompt
from lumintelligence.storage import SqliteVecStore, VectorStore

__all__ = [
    "Config",
    "GraphStore",
    "LumIntelligence",
    "OpenAIConfig",
    "SqliteGraphStore",
    "SqliteVecStore",
    "VectorStore",
    "__version__",
    "load_prompt",
]
