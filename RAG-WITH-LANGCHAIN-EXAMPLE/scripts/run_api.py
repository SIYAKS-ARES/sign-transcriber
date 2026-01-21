#!/usr/bin/env python3
"""
API Sunucu Baslatma Script'i
"""

import sys
from pathlib import Path

# Proje root'una path ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from src.config import settings


def main():
    """API sunucuyu baslat"""
    print("=" * 60)
    print("LGS RAG API Sunucusu")
    print("=" * 60)
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Docs: http://localhost:{settings.api_port}/docs")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

