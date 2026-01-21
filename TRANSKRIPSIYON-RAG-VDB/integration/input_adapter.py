"""
Input Adapter for Sign-Transcriber Ecosystem
=============================================
Handles various input formats from the transcription pipeline.
"""

from dataclasses import dataclass
from typing import Union, List, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TranscriptionInput:
    """Standardized transcription input."""
    glosses: List[str]      # ['OKUL', 'GITMEK', 'ISTEMEK']
    raw_string: str         # "OKUL GITMEK ISTEMEK"
    confidence: float       # Model confidence (0-1)
    source: str             # Source of input ("model", "manual", "api")
    
    def __str__(self) -> str:
        return self.raw_string


def adapt_input(raw_input: Union[str, List[str], Dict[str, Any]]) -> TranscriptionInput:
    """
    Convert various input formats to standardized TranscriptionInput.
    
    Supported formats:
    - str: "OKUL GITMEK ISTEMEK"
    - List[str]: ['OKUL', 'GITMEK', 'ISTEMEK']
    - dict: {'glosses': [...], 'confidence': 0.85, 'source': 'model'}
    
    Args:
        raw_input: Input in any supported format
        
    Returns:
        Standardized TranscriptionInput
        
    Raises:
        TypeError: If input format is not supported
        ValueError: If input is empty or invalid
    """
    if isinstance(raw_input, str):
        if not raw_input.strip():
            raise ValueError("Empty string input")
        
        glosses = raw_input.upper().split()
        return TranscriptionInput(
            glosses=glosses,
            raw_string=" ".join(glosses),
            confidence=1.0,
            source="manual",
        )
    
    elif isinstance(raw_input, list):
        if not raw_input:
            raise ValueError("Empty list input")
        
        glosses = [str(g).upper() for g in raw_input]
        return TranscriptionInput(
            glosses=glosses,
            raw_string=" ".join(glosses),
            confidence=1.0,
            source="manual",
        )
    
    elif isinstance(raw_input, dict):
        # Handle dict input with various possible keys
        glosses = raw_input.get('glosses', [])
        
        # Also check for 'transcription' or 'gloss' keys
        if not glosses:
            if 'transcription' in raw_input:
                trans = raw_input['transcription']
                glosses = trans.upper().split() if isinstance(trans, str) else [str(g).upper() for g in trans]
            elif 'gloss' in raw_input:
                gloss = raw_input['gloss']
                glosses = gloss.upper().split() if isinstance(gloss, str) else [str(g).upper() for g in gloss]
        else:
            glosses = [str(g).upper() for g in glosses]
        
        if not glosses:
            raise ValueError("Dict input has no valid glosses")
        
        return TranscriptionInput(
            glosses=glosses,
            raw_string=" ".join(glosses),
            confidence=float(raw_input.get('confidence', 1.0)),
            source=raw_input.get('source', 'api'),
        )
    
    elif isinstance(raw_input, TranscriptionInput):
        # Already a TranscriptionInput, just return it
        return raw_input
    
    else:
        raise TypeError(f"Unsupported input type: {type(raw_input).__name__}")


def normalize_transcription(transcription: str) -> str:
    """
    Normalize a transcription string.
    
    - Convert to uppercase
    - Normalize whitespace
    - Handle special characters (^ for compound words)
    
    Args:
        transcription: Raw transcription string
        
    Returns:
        Normalized transcription
    """
    if not transcription:
        return ""
    
    # Convert to uppercase
    result = transcription.upper()
    
    # Normalize whitespace
    import re
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


if __name__ == "__main__":
    # Test the input adapter
    print("Testing Input Adapter...")
    print("=" * 60)
    
    # Test string input
    result = adapt_input("okul gitmek istemek")
    print(f"String input: {result}")
    print(f"  Glosses: {result.glosses}")
    print(f"  Confidence: {result.confidence}")
    
    # Test list input
    result = adapt_input(['OKUL', 'GITMEK', 'ISTEMEK'])
    print(f"\nList input: {result}")
    
    # Test dict input
    result = adapt_input({
        'glosses': ['OKUL', 'GITMEK'],
        'confidence': 0.85,
        'source': 'model'
    })
    print(f"\nDict input: {result}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Source: {result.source}")
    
    # Test dict with 'transcription' key
    result = adapt_input({'transcription': 'BEN OKUL GITMEK'})
    print(f"\nDict with transcription: {result}")
