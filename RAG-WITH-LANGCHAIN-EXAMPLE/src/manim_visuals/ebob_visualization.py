"""
R-09: Manim Gorsel Uretim Modulu
EBOB/EKOK kavramlarinin gorsellestirmesi
"""

from manim import *


class EBOBVisualization(Scene):
    """
    EBOB kavramini gorsel olarak aciklar
    """

    def construct(self):
        # Baslik
        title = Text("EBOB Nedir?", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Iki sayi
        num1 = Text("24", font_size=72, color=BLUE)
        num2 = Text("36", font_size=72, color=GREEN)
        nums = VGroup(num1, num2).arrange(RIGHT, buff=2)

        self.play(FadeIn(nums))
        self.wait(1)

        # Carpanlar
        factors1 = Text("24 = 2 x 2 x 2 x 3", font_size=36).next_to(num1, DOWN)
        factors2 = Text("36 = 2 x 2 x 3 x 3", font_size=36).next_to(num2, DOWN)

        self.play(Write(factors1), Write(factors2))
        self.wait(2)

        # Ortak carpanlar vurgula
        common = Text("Ortak: 2 x 2 x 3 = 12", font_size=42, color=YELLOW)
        common.to_edge(DOWN)

        self.play(Write(common))

        # EBOB sonucu
        result = Text("EBOB(24, 36) = 12", font_size=48, color=RED)
        result.next_to(common, UP)

        self.play(FadeIn(result, scale=1.5), Flash(result, color=RED))
        self.wait(2)


class EKOKVisualization(Scene):
    """
    EKOK kavramini gorsel olarak aciklar
    """

    def construct(self):
        # Baslik
        title = Text("EKOK Nedir?", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Iki sayi
        num1 = Text("4", font_size=72, color=BLUE)
        num2 = Text("6", font_size=72, color=GREEN)
        nums = VGroup(num1, num2).arrange(RIGHT, buff=3)

        self.play(FadeIn(nums))
        self.wait(1)

        # Katlari goster
        multiples1_text = Text("4, 8, 12, 16, 20, 24...", font_size=32, color=BLUE)
        multiples2_text = Text("6, 12, 18, 24, 30...", font_size=32, color=GREEN)

        multiples1_text.next_to(num1, DOWN, buff=0.5)
        multiples2_text.next_to(num2, DOWN, buff=0.5)

        self.play(Write(multiples1_text), Write(multiples2_text))
        self.wait(2)

        # Ortak katlar
        common = Text("Ortak Katlar: 12, 24...", font_size=36, color=YELLOW)
        common.to_edge(DOWN, buff=1.5)

        self.play(Write(common))
        self.wait(1)

        # EKOK sonucu
        result = Text("EKOK(4, 6) = 12", font_size=48, color=RED)
        result.next_to(common, UP)

        self.play(FadeIn(result, scale=1.5), Flash(result, color=RED))
        self.wait(2)


class CarpanAgaci(Scene):
    """
    Bir sayinin carpan agacini gosterir
    """

    def construct(self):
        # 60 sayisinin carpan agaci
        title = Text("60'in Asal Carpanlari", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))

        # Agac yapisini daha basit goster
        root = Text("60", font_size=40)
        root.move_to(UP * 2)

        level1_left = Text("2", font_size=36, color=RED)
        level1_right = Text("30", font_size=36)
        level1_left.move_to(LEFT * 2 + UP * 0.5)
        level1_right.move_to(RIGHT * 2 + UP * 0.5)

        level2_left = Text("2", font_size=36, color=RED)
        level2_right = Text("15", font_size=36)
        level2_left.move_to(RIGHT * 1 + DOWN * 0.5)
        level2_right.move_to(RIGHT * 3 + DOWN * 0.5)

        level3_left = Text("3", font_size=36, color=RED)
        level3_right = Text("5", font_size=36, color=RED)
        level3_left.move_to(RIGHT * 2 + DOWN * 1.5)
        level3_right.move_to(RIGHT * 4 + DOWN * 1.5)

        # Cizgiler
        line1 = Line(root.get_bottom(), level1_left.get_top())
        line2 = Line(root.get_bottom(), level1_right.get_top())
        line3 = Line(level1_right.get_bottom(), level2_left.get_top())
        line4 = Line(level1_right.get_bottom(), level2_right.get_top())
        line5 = Line(level2_right.get_bottom(), level3_left.get_top())
        line6 = Line(level2_right.get_bottom(), level3_right.get_top())

        # Animasyonlar
        self.play(Create(root))
        self.wait(0.5)

        self.play(Create(line1), Create(line2))
        self.play(Create(level1_left), Create(level1_right))
        self.wait(0.5)

        self.play(Create(line3), Create(line4))
        self.play(Create(level2_left), Create(level2_right))
        self.wait(0.5)

        self.play(Create(line5), Create(line6))
        self.play(Create(level3_left), Create(level3_right))
        self.wait(1)

        # Sonuc
        result = Text("60 = 2 x 2 x 3 x 5", font_size=36)
        result.to_edge(DOWN)
        self.play(Write(result))
        self.wait(2)


class TabloGorsel(Scene):
    """
    Tablo tipinde gorsel uretir
    """

    def __init__(self, headers: list = None, data: list = None, **kwargs):
        super().__init__(**kwargs)
        self.headers = headers or ["Urun", "Miktar"]
        self.data = data or [["Elma", "24"], ["Armut", "36"]]

    def construct(self):
        # Tablo basligi
        title = Text("Urun Tablosu", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Tablo olustur
        table_data = [self.headers] + self.data
        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 2},
        ).scale(0.7)

        self.play(Create(table))
        self.wait(2)


class SematikGorsel(Scene):
    """
    Sematik tipinde gorsel uretir
    (Ok, kutu, iliskiler)
    """

    def construct(self):
        # Baslik
        title = Text("EBOB Problemi", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Kutular
        box1 = Rectangle(width=2, height=1.5, color=BLUE)
        box1.shift(LEFT * 3)
        label1 = Text("24 elma", font_size=24).move_to(box1)

        box2 = Rectangle(width=2, height=1.5, color=GREEN)
        box2.shift(RIGHT * 3)
        label2 = Text("36 armut", font_size=24).move_to(box2)

        self.play(Create(box1), Write(label1))
        self.play(Create(box2), Write(label2))

        # Ok ve orta kutu
        arrow1 = Arrow(box1.get_right(), ORIGIN + LEFT * 0.5, color=YELLOW)
        arrow2 = Arrow(box2.get_left(), ORIGIN + RIGHT * 0.5, color=YELLOW)

        center_box = Rectangle(width=2.5, height=1.5, color=RED)
        center_label = Text("EBOB = 12", font_size=24).move_to(center_box)

        self.play(Create(arrow1), Create(arrow2))
        self.play(Create(center_box), Write(center_label))

        # Sonuc
        result = Text("Her sepete 12 tane esit dagilir", font_size=28)
        result.to_edge(DOWN)
        self.play(Write(result))
        self.wait(2)


# Render fonksiyonu
def render_visualization(
    scene_class,
    output_path: str = None,
    quality: str = "l",
    **scene_kwargs,
):
    """
    Manim sahnesini render et
    
    Args:
        scene_class: Render edilecek Scene sinifi
        output_path: Cikti dosya yolu
        quality: Kalite (l=low, m=medium, h=high)
        **scene_kwargs: Scene constructor argumanlari
        
    Returns:
        Cikti dosya yolu
    """
    from manim import config

    # Kalite ayarlari
    quality_settings = {
        "l": {"pixel_width": 854, "pixel_height": 480, "frame_rate": 15},
        "m": {"pixel_width": 1280, "pixel_height": 720, "frame_rate": 30},
        "h": {"pixel_width": 1920, "pixel_height": 1080, "frame_rate": 60},
    }

    settings = quality_settings.get(quality, quality_settings["l"])

    config.pixel_width = settings["pixel_width"]
    config.pixel_height = settings["pixel_height"]
    config.frame_rate = settings["frame_rate"]

    if output_path:
        config.output_file = output_path

    # Render
    scene = scene_class(**scene_kwargs)
    scene.render()

    return config.output_file

