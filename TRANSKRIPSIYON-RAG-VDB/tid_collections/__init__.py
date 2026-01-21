"""ChromaDB collection management for TID_Sozluk and TID_Hafiza."""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "SozlukCollection":
        from .sozluk_collection import SozlukCollection
        return SozlukCollection
    elif name == "HafizaCollection":
        from .hafiza_collection import HafizaCollection
        return HafizaCollection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SozlukCollection", "HafizaCollection"]
