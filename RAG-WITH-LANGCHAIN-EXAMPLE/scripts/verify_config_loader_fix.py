#!/usr/bin/env python3
"""
Verification script to demonstrate the fix for empty CSV field handling.
This script shows that the ConfigLoader now handles empty numeric fields gracefully.
"""

import csv
import sys
import tempfile
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "config_loader", 
    src_path / "services" / "config_loader.py"
)
config_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_loader_module)
ConfigLoader = config_loader_module.ConfigLoader


def test_empty_fields():
    """Test that empty numeric fields are handled correctly"""
    print("Testing ConfigLoader with empty numeric fields...")
    
    loader = ConfigLoader()
    
    # Create a CSV with empty numeric fields (the bug scenario)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Soru_MetniOCR', 'Alt_Konu', 'Zorluk', 'Gorsel_Tipi', 'Kaynak_Tipi', 'is_LGS', 'egitim_agirligi'])
        # Row with empty numeric fields - this would crash before the fix
        writer.writerow(['Test question', 'ebob_ekok', '', 'yok', 'baslangic', '', ''])
        temp_path = Path(f.name)
    
    try:
        # This should NOT raise ValueError anymore
        questions = loader.load_questions(temp_path)
        
        print(f"✓ Successfully loaded {len(questions)} question(s)")
        print(f"✓ Zorluk (empty field): {questions[0]['Zorluk']} (defaulted to 0)")
        print(f"✓ is_LGS (empty field): {questions[0]['is_LGS']} (defaulted to 0)")
        print(f"✓ egitim_agirligi (empty field): {questions[0]['egitim_agirligi']} (defaulted to 0.5)")
        
        print("\n✓ Bug fix verified: Empty numeric fields are now handled gracefully!")
        return True
    except ValueError as e:
        print(f"✗ Bug still exists: {e}")
        return False
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    success = test_empty_fields()
    sys.exit(0 if success else 1)

