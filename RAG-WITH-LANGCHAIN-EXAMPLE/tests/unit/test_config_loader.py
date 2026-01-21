"""
Unit Tests for ConfigLoader
"""

import csv
import tempfile
from pathlib import Path

import pytest

from services.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader"""

    def test_load_questions_handles_empty_numeric_fields(self):
        """Test that empty numeric fields in CSV are handled gracefully"""
        loader = ConfigLoader()
        
        # Create a temporary CSV with empty numeric fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Soru_MetniOCR', 'Alt_Konu', 'Zorluk', 'Gorsel_Tipi', 'Kaynak_Tipi', 'is_LGS', 'egitim_agirligi'])
            writer.writerow(['Test question', 'ebob_ekok', '', 'yok', 'baslangic', '', ''])
            temp_path = Path(f.name)
        
        try:
            questions = loader.load_questions(temp_path)
            
            assert len(questions) == 1
            assert questions[0]['Soru_MetniOCR'] == 'Test question'
            assert questions[0]['Zorluk'] == 0  # Should default to 0 for empty string
            assert questions[0]['is_LGS'] == 0  # Should default to 0 for empty string
            assert questions[0]['egitim_agirligi'] == 0.5  # Should default to 0.5 for empty string
        finally:
            temp_path.unlink()

    def test_load_questions_handles_whitespace_numeric_fields(self):
        """Test that whitespace-only numeric fields are handled gracefully"""
        loader = ConfigLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Soru_MetniOCR', 'Alt_Konu', 'Zorluk', 'Gorsel_Tipi', 'Kaynak_Tipi', 'is_LGS', 'egitim_agirligi'])
            writer.writerow(['Test question', 'ebob_ekok', '   ', 'yok', 'baslangic', '  ', '  '])
            temp_path = Path(f.name)
        
        try:
            questions = loader.load_questions(temp_path)
            
            assert len(questions) == 1
            assert questions[0]['Zorluk'] == 0
            assert questions[0]['is_LGS'] == 0
            assert questions[0]['egitim_agirligi'] == 0.5
        finally:
            temp_path.unlink()

    def test_load_questions_handles_valid_numeric_fields(self):
        """Test that valid numeric fields are parsed correctly"""
        loader = ConfigLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Soru_MetniOCR', 'Alt_Konu', 'Zorluk', 'Gorsel_Tipi', 'Kaynak_Tipi', 'is_LGS', 'egitim_agirligi'])
            writer.writerow(['Test question', 'ebob_ekok', '3', 'yok', 'baslangic', '1', '0.8'])
            temp_path = Path(f.name)
        
        try:
            questions = loader.load_questions(temp_path)
            
            assert len(questions) == 1
            assert questions[0]['Zorluk'] == 3
            assert questions[0]['is_LGS'] == 1
            assert questions[0]['egitim_agirligi'] == 0.8
        finally:
            temp_path.unlink()

    def test_load_questions_handles_invalid_numeric_fields(self):
        """Test that invalid numeric fields default gracefully"""
        loader = ConfigLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Soru_MetniOCR', 'Alt_Konu', 'Zorluk', 'Gorsel_Tipi', 'Kaynak_Tipi', 'is_LGS', 'egitim_agirligi'])
            writer.writerow(['Test question', 'ebob_ekok', 'invalid', 'yok', 'baslangic', 'not_a_number', 'also_invalid'])
            temp_path = Path(f.name)
        
        try:
            questions = loader.load_questions(temp_path)
            
            assert len(questions) == 1
            # Should default to 0 for invalid int values
            assert questions[0]['Zorluk'] == 0
            assert questions[0]['is_LGS'] == 0
            # Should default to 0.5 for invalid float values
            assert questions[0]['egitim_agirligi'] == 0.5
        finally:
            temp_path.unlink()

    def test_load_configs_valid_json(self, sample_configs_path):
        """Test loading valid configs.json"""
        loader = ConfigLoader()
        configs = loader.load_configs(sample_configs_path)
        
        assert configs is not None
        assert "combinations" in configs
        assert len(configs["combinations"]) > 0

    def test_load_configs_missing_file(self):
        """Test that missing configs file raises FileNotFoundError"""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_configs("nonexistent_file.json")

    def test_load_questions_missing_file(self):
        """Test that missing questions file raises FileNotFoundError"""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_questions("nonexistent_file.csv")

