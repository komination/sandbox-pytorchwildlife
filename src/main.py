from PytorchWildlife.models.detection import MegaDetectorV6Apache

# Apache RT‑DETR
detector = MegaDetectorV6Apache(
    device="cpu",
    pretrained=True,
    version="MDV6-apa-rtdetr-e"
)