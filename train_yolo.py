#!/usr/bin/env python3
"""
Skrypt treningowy YOLOv8n do rozpoznawania czerwonej piłki.
"""

from ultralytics import YOLO
from pathlib import Path


# Konfiguracja
DATA_YAML = Path("red_ball_augmented/data.yaml")
MODEL_NAME = "yolov8n.pt"  # Pretrenowany model nano (najszybszy)
OUTPUT_DIR = Path("runs/detect/red_ball")

# Parametry treningu
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 512  # Dopasowane do rozmiaru obrazów w datasecie
PATIENCE = 20   # Early stopping - zatrzymaj jeśli brak poprawy przez 20 epok


def main():
    print("=" * 60)
    print("Trening YOLOv8n - Rozpoznawanie czerwonej piłki")
    print("=" * 60)

    # Sprawdź czy plik konfiguracyjny istnieje
    if not DATA_YAML.exists():
        print(f"\nBŁĄD: Nie znaleziono pliku {DATA_YAML}")
        print("Najpierw uruchom: python augment_dataset.py")
        return

    print(f"\n1. Konfiguracja datasetu: {DATA_YAML}")
    print(f"2. Model bazowy: {MODEL_NAME}")
    print(f"3. Epoki: {EPOCHS}")
    print(f"4. Batch size: {BATCH_SIZE}")
    print(f"5. Rozmiar obrazu: {IMG_SIZE}")

    # Wczytaj pretrenowany model
    print("\n" + "-" * 60)
    print("Ładowanie modelu...")
    model = YOLO(MODEL_NAME)

    # Rozpocznij trening
    print("\nRozpoczynam trening...")
    print("-" * 60)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        save=True,
        save_period=10,          # Zapisuj checkpoint co 10 epok
        project="runs/detect",
        name="red_ball",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        device="cpu",              # GPU 0 (zmień na "cpu" jeśli brak GPU)
        workers=4,
        # Augmentacje wbudowane w YOLO (delikatne - mamy już własne)
        hsv_h=0.01,              # Minimalna zmiana hue (zachowujemy czerwień)
        hsv_s=0.3,               # Nasycenie
        hsv_v=0.3,               # Jasność
        degrees=15,              # Rotacja
        translate=0.1,           # Przesunięcie
        scale=0.3,               # Skalowanie
        flipud=0.0,              # Bez flip pionowego
        fliplr=0.5,              # Flip poziomy
        mosaic=0.5,              # Mosaic augmentation
        mixup=0.0,               # Bez mixup
    )

    print("\n" + "=" * 60)
    print("TRENING ZAKOŃCZONY!")
    print("=" * 60)

    # Ewaluacja na zbiorze testowym
    print("\nEwaluacja na zbiorze testowym...")
    metrics = model.val(data=str(DATA_YAML), split="test")

    print("\n" + "-" * 60)
    print("WYNIKI NA ZBIORZE TESTOWYM:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("-" * 60)

    # Ścieżka do najlepszego modelu
    best_model_path = Path("runs/detect/red_ball/weights/best.pt")
    print(f"\nNajlepszy model zapisany w: {best_model_path}")

    # Eksport do różnych formatów (opcjonalnie)
    print("\nEksport modelu do formatu ONNX...")
    model.export(format="onnx", imgsz=IMG_SIZE)
    print("Model ONNX zapisany.")

    print("\n" + "=" * 60)
    print("GOTOWE!")
    print(f"Użyj modelu: YOLO('{best_model_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
