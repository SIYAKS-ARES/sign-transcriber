"""
R-09: Gorsel Uretim Servisi
Uretilen sorular icin Manim gorselleri olusturur
"""

import subprocess
from pathlib import Path
from typing import Optional

from ..config import settings


class VisualGenerator:
    """
    Uretilen sorular icin Manim gorselleri olusturur
    
    Desteklenen gorsel tipleri:
    - tablo: Tablo gorseli
    - sematik: Sema/diagram
    - geometrik_sekil: Geometrik sekiller
    - resimli: Illustrasyon
    """

    GORSEL_TIPI_SCENES = {
        "tablo": "TabloGorsel",
        "sematik": "SematikGorsel",
        "geometrik_sekil": "SematikGorsel",  # Ayni scene kullan
    }

    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: Cikti dizini
        """
        self.output_dir = Path(output_dir or settings.manim_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality = settings.manim_quality

    def generate_visual(self, question: dict) -> Optional[str]:
        """
        Soru icin gorsel uret
        
        Args:
            question: Uretilen soru dict (GeneratedQuestion.model_dump())
            
        Returns:
            Gorsel dosya yolu veya None
        """
        gorsel_tipi = question.get("gorsel_tipi", "yok")

        if gorsel_tipi == "yok":
            return None

        scene_name = self.GORSEL_TIPI_SCENES.get(gorsel_tipi)
        if not scene_name:
            print(f"Desteklenmeyen gorsel tipi: {gorsel_tipi}")
            return None

        # Dosya adi
        question_id = question.get("id", "temp")
        output_file = self.output_dir / f"soru_{question_id}.mp4"

        try:
            # Manim render komutu
            result = subprocess.run(
                [
                    "manim",
                    "render",
                    f"-q{self.quality}",
                    str(Path(__file__).parent / "ebob_visualization.py"),
                    scene_name,
                    "-o",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=60,  # 1 dakika timeout
            )

            if result.returncode == 0:
                print(f"Gorsel olusturuldu: {output_file}")
                return str(output_file)
            else:
                print(f"Manim hatasi: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("Gorsel uretimi zaman asimina ugradi")
            return None
        except FileNotFoundError:
            print("Manim yuklu degil. 'pip install manim' ile yukleyin.")
            return None
        except Exception as e:
            print(f"Gorsel uretim hatasi: {e}")
            return None

    def generate_ebob_visual(
        self,
        num1: int,
        num2: int,
        output_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        EBOB gorseli uret
        
        Args:
            num1: Birinci sayi
            num2: Ikinci sayi
            output_name: Cikti dosya adi
            
        Returns:
            Gorsel dosya yolu veya None
        """
        output_file = self.output_dir / (output_name or f"ebob_{num1}_{num2}.mp4")

        try:
            result = subprocess.run(
                [
                    "manim",
                    "render",
                    f"-q{self.quality}",
                    str(Path(__file__).parent / "ebob_visualization.py"),
                    "EBOBVisualization",
                    "-o",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return str(output_file)
            return None

        except Exception as e:
            print(f"EBOB gorsel hatasi: {e}")
            return None

    def generate_ekok_visual(
        self,
        num1: int,
        num2: int,
        output_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        EKOK gorseli uret
        """
        output_file = self.output_dir / (output_name or f"ekok_{num1}_{num2}.mp4")

        try:
            result = subprocess.run(
                [
                    "manim",
                    "render",
                    f"-q{self.quality}",
                    str(Path(__file__).parent / "ebob_visualization.py"),
                    "EKOKVisualization",
                    "-o",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return str(output_file)
            return None

        except Exception as e:
            print(f"EKOK gorsel hatasi: {e}")
            return None

    def export_for_unity(self, visual_path: str, unity_assets_path: str) -> bool:
        """
        Manim gorselini Unity Assets klasorune kopyala
        
        Bu gorsel, AR uygulamasinda soru gosterimi icin kullanilabilir.
        
        Args:
            visual_path: Kaynak gorsel dosya yolu
            unity_assets_path: Unity Assets dizini
            
        Returns:
            Basarili ise True
        """
        import shutil

        try:
            source = Path(visual_path)
            if not source.exists():
                print(f"Kaynak dosya bulunamadi: {visual_path}")
                return False

            dest_dir = Path(unity_assets_path) / "GeneratedVisuals"
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest = dest_dir / source.name
            shutil.copy(source, dest)

            print(f"Unity'ye aktarildi: {dest}")
            return True

        except Exception as e:
            print(f"Unity export hatasi: {e}")
            return False

    def list_generated_visuals(self) -> list[str]:
        """
        Olusturulan gorselleri listele
        
        Returns:
            Dosya yollari listesi
        """
        visuals = []
        for ext in ["*.mp4", "*.gif", "*.png"]:
            visuals.extend(str(p) for p in self.output_dir.glob(ext))
        return sorted(visuals)

    def cleanup(self, older_than_days: int = 7):
        """
        Eski gorselleri temizle
        
        Args:
            older_than_days: Bu gunlerden eski dosyalari sil
        """
        import time

        cutoff = time.time() - (older_than_days * 24 * 60 * 60)

        deleted = 0
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                deleted += 1

        print(f"{deleted} eski gorsel silindi")

