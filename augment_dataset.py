#!/usr/bin/env python3
"""
Skrypt do augmentacji datasetu COCO z czerwoną piłką.
Dodaje losowe elementy tła i stosuje różne transformacje.
Tworzy podział na train/val/test.
"""

import json
import os
import random
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO
import shutil
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# Konfiguracja
INPUT_DIR = Path("red_ball.v1i.coco/train")
OUTPUT_BASE_DIR = Path("red_ball_augmented")
ANNOTATIONS_FILE = INPUT_DIR / "_annotations.coco.json"

# Liczba augmentacji na obraz
AUGMENTATIONS_PER_IMAGE = 3

# Podział datasetu (proporcje)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Przykładowe URL-e do pobrania losowych obrazów (Lorem Picsum - darmowe losowe zdjęcia)
RANDOM_IMAGE_URLS = [
    "https://picsum.photos/200/200",  # Losowe zdjęcie 200x200
]


def download_random_images(count: int = 20) -> List[Image.Image]:
    """Pobiera losowe obrazy z internetu."""
    images = []
    print(f"Pobieranie {count} losowych obrazów...")

    for i in range(count):
        try:
            # Używamy Lorem Picsum - darmowe losowe zdjęcia
            response = requests.get(
                f"https://picsum.photos/{random.randint(100, 300)}/{random.randint(100, 300)}",
                timeout=10
            )
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
                print(f"  Pobrano obraz {i+1}/{count}")
        except Exception as e:
            print(f"  Błąd pobierania obrazu {i+1}: {e}")

    return images


def create_local_random_patches(count: int = 30) -> List[Image.Image]:
    """Tworzy losowe kolorowe patche jako alternatywę dla pobierania."""
    patches = []

    for _ in range(count):
        # Losowy rozmiar
        w = random.randint(50, 150)
        h = random.randint(50, 150)

        # Losowy typ patcha
        patch_type = random.choice(['solid', 'gradient', 'noise', 'pattern'])

        if patch_type == 'solid':
            # Jednolity kolor (ale nie czerwony - unikamy mylenia z piłką)
            color = (
                random.randint(0, 150),  # Ograniczony czerwony
                random.randint(0, 255),
                random.randint(0, 255)
            )
            img = Image.new('RGB', (w, h), color)

        elif patch_type == 'gradient':
            # Gradient
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    arr[y, x] = [
                        int(255 * x / w) if random.random() > 0.5 else random.randint(0, 150),
                        int(255 * y / h) if random.random() > 0.5 else random.randint(0, 255),
                        random.randint(100, 255)
                    ]
            img = Image.fromarray(arr)

        elif patch_type == 'noise':
            # Szum
            arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            # Zmniejsz czerwony kanał żeby nie mylić z piłką
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.6, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        else:  # pattern
            # Wzór (paski/kratka)
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            stripe_width = random.randint(5, 20)
            color1 = [random.randint(0, 150), random.randint(0, 255), random.randint(0, 255)]
            color2 = [random.randint(0, 150), random.randint(0, 255), random.randint(0, 255)]

            for y in range(h):
                for x in range(w):
                    if (x // stripe_width + y // stripe_width) % 2 == 0:
                        arr[y, x] = color1
                    else:
                        arr[y, x] = color2
            img = Image.fromarray(arr)

        patches.append(img)

    return patches


def get_ball_mask(image_size: Tuple[int, int], annotations: List[Dict]) -> np.ndarray:
    """Tworzy maskę obszaru piłki (gdzie NIE nakładać patchy)."""
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Dodaj większy margines wokół piłki - patche nie mogą jej dotykać
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_size[0], x + w + margin)
        y2 = min(image_size[1], y + h + margin)
        mask[y1:y2, x1:x2] = 255

    return mask


def overlay_random_patches(
    image: Image.Image,
    patches: List[Image.Image],
    ball_mask: np.ndarray,
    num_patches: int = 3
) -> Image.Image:
    """Nakłada losowe patche na obraz, omijając obszar piłki."""
    result = image.copy()
    img_w, img_h = image.size

    for _ in range(num_patches):
        if not patches:
            break

        patch = random.choice(patches).copy()

        # Losowa transformacja patcha
        if random.random() > 0.5:
            patch = patch.rotate(random.randint(0, 360), expand=True)
        if random.random() > 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

        # Zmień rozmiar patcha
        scale = random.uniform(0.3, 1.5)
        new_size = (int(patch.width * scale), int(patch.height * scale))
        new_size = (max(20, min(new_size[0], img_w // 2)),
                    max(20, min(new_size[1], img_h // 2)))
        patch = patch.resize(new_size, Image.LANCZOS)

        # Znajdź miejsce gdzie można nałożyć (NIGDY na piłkę)
        max_attempts = 50
        placed = False
        for _ in range(max_attempts):
            x = random.randint(0, max(0, img_w - patch.width))
            y = random.randint(0, max(0, img_h - patch.height))

            # Sprawdź czy patch w ogóle nie nachodzi na piłkę (maska musi być całkowicie zerowa)
            patch_area = ball_mask[y:y+patch.height, x:x+patch.width]
            if patch_area.size > 0 and np.max(patch_area) <5:  # Żaden piksel nie nachodzi na piłkę
                # Nałóż z losową przezroczystością
                alpha = random.uniform(0.3, 0.8)

                # Wytnij obszar z oryginalnego obrazu
                region = result.crop((x, y, x + patch.width, y + patch.height))

                # Blend
                blended = Image.blend(region, patch.resize(region.size), alpha)
                result.paste(blended, (x, y))
                placed = True
                break

        # Jeśli nie udało się znaleźć miejsca, pomijamy ten patch
        if not placed:
            continue

    return result


def apply_augmentation(image: Image.Image, aug_type: str) -> Image.Image:
    """Stosuje pojedynczą augmentację - zachowuje czerwień piłki."""
    result = image.copy()

    if aug_type == 'brightness':
        # Delikatna zmiana jasności
        factor = random.uniform(0.75, 1.25)
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(factor)

    elif aug_type == 'contrast':
        # Delikatna zmiana kontrastu
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(factor)

    elif aug_type == 'saturation':
        # Tylko zwiększenie nasycenia (nie zmniejszamy - zachowujemy czerwień)
        factor = random.uniform(0.9, 1.3)
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(factor)

    elif aug_type == 'blur':
        # Lekkie rozmycie
        result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    elif aug_type == 'sharpen':
        result = result.filter(ImageFilter.SHARPEN)

    elif aug_type == 'noise':
        # Delikatny szum - nie za mocny żeby nie zmienić kolorów
        arr = np.array(result)
        noise = np.random.normal(0, random.randint(3, 10), arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr)

    return result


def augment_image(
    image: Image.Image,
    patches: List[Image.Image],
    ball_mask: np.ndarray
) -> Image.Image:
    """Stosuje pełną augmentację na obrazie."""
    result = image.copy()

    # 1. Nałóż losowe patche (50% szans)
    if random.random() > 0.5:
        num_patches = random.randint(1, 4)
        result = overlay_random_patches(result, patches, ball_mask, num_patches)

    # 2. Stosuj losowe augmentacje (bez zmiany odcieni - zachowujemy czerwień piłki)
    augmentations = ['brightness', 'contrast', 'saturation', 'blur', 'sharpen', 'noise']
    num_augs = random.randint(1, 3)
    selected_augs = random.sample(augmentations, num_augs)

    for aug in selected_augs:
        result = apply_augmentation(result, aug)

    return result


def update_annotation_for_new_image(
    original_ann: Dict,
    new_image_id: int,
    new_ann_id: int
) -> Dict:
    """Tworzy nową adnotację dla augmentowanego obrazu."""
    new_ann = original_ann.copy()
    new_ann['id'] = new_ann_id
    new_ann['image_id'] = new_image_id
    return new_ann


def save_yolo_format(images_data: List[Dict], annotations_data: List[Dict],
                     output_dir: Path, coco_data: Dict):
    """Zapisuje dane w formacie YOLO (txt z normalizowanymi bbox)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Mapowanie image_id -> annotations
    img_to_anns = {}
    for ann in annotations_data:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    for img_info in images_data:
        # Kopiuj/przenieś obraz
        src_path = img_info.get('_src_path')
        if src_path and Path(src_path).exists():
            dst_path = images_dir / img_info['file_name']
            shutil.copy(src_path, dst_path)

        # Zapisz etykiety w formacie YOLO
        img_w = img_info['width']
        img_h = img_info['height']
        anns = img_to_anns.get(img_info['id'], [])

        label_filename = img_info['file_name'].rsplit('.', 1)[0] + '.txt'
        label_path = labels_dir / label_filename

        with open(label_path, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height] w COCO
                # Konwersja do YOLO: [class_id, x_center, y_center, width, height] (znormalizowane)
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w_norm = bbox[2] / img_w
                h_norm = bbox[3] / img_h
                # Klasa 0 = ball
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def main():
    print("=" * 60)
    print("Augmentacja datasetu COCO z czerwoną piłką")
    print("Podział na train/val/test w formacie YOLO")
    print("=" * 60)

    # Wczytaj adnotacje
    print("\n1. Wczytywanie adnotacji...")
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco_data = json.load(f)

    print(f"   Znaleziono {len(coco_data['images'])} obrazów")
    print(f"   Znaleziono {len(coco_data['annotations'])} adnotacji")

    # Stwórz katalogi wyjściowe
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Pobierz lub stwórz losowe patche
    print("\n2. Przygotowywanie losowych patchy...")
    try:
        patches = download_random_images(15)
        if len(patches) < 5:
            print("   Mało pobranych obrazów, generuję lokalne patche...")
            patches.extend(create_local_random_patches(20))
    except Exception as e:
        print(f"   Błąd pobierania: {e}")
        print("   Generuję lokalne patche...")
        patches = create_local_random_patches(30)

    print(f"   Przygotowano {len(patches)} patchy")

    # Mapowanie image_id -> annotations
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Przygotuj wszystkie obrazy (oryginalne + augmentacje)
    print("\n3. Tworzenie augmentacji...")
    all_images = []  # Lista (img_info, annotations, src_path)
    next_image_id = max(img['id'] for img in coco_data['images']) + 1
    next_ann_id = max(ann['id'] for ann in coco_data['annotations']) + 1

    total_images = len(coco_data['images'])
    for idx, img_info in enumerate(coco_data['images']):
        img_path = INPUT_DIR / img_info['file_name']

        if not img_path.exists():
            print(f"   UWAGA: Brak pliku {img_path}")
            continue

        # Wczytaj obraz
        image = Image.open(img_path).convert("RGB")
        img_anns = img_to_anns.get(img_info['id'], [])

        # Stwórz maskę piłki
        ball_mask = get_ball_mask(image.size, img_anns)

        # Dodaj oryginalny obraz
        orig_info = img_info.copy()
        orig_info['_src_path'] = str(img_path)
        all_images.append((orig_info, [ann.copy() for ann in img_anns]))

        # Twórz augmentacje
        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            aug_image = augment_image(image, patches, ball_mask)

            # Nowa nazwa pliku
            base_name = img_info['file_name'].rsplit('.', 1)[0]
            ext = img_info['file_name'].rsplit('.', 1)[1]
            aug_filename = f"{base_name}_aug{aug_idx}.{ext}"

            # Zapisz tymczasowo augmentowany obraz
            temp_dir = OUTPUT_BASE_DIR / "temp"
            temp_dir.mkdir(exist_ok=True)
            aug_path = temp_dir / aug_filename
            aug_image.save(aug_path, quality=95)

            # Nowe info o obrazie
            new_img_info = img_info.copy()
            new_img_info['id'] = next_image_id
            new_img_info['file_name'] = aug_filename
            new_img_info['_src_path'] = str(aug_path)

            # Nowe adnotacje
            new_anns = []
            for ann in img_anns:
                new_ann = update_annotation_for_new_image(ann, next_image_id, next_ann_id)
                new_anns.append(new_ann)
                next_ann_id += 1

            all_images.append((new_img_info, new_anns))
            next_image_id += 1

        print(f"   Przetworzono {idx+1}/{total_images}: {img_info['file_name']}")

    # Podziel na train/val/test
    print("\n4. Podział na train/val/test...")
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_data = all_images[:n_train]
    val_data = all_images[n_train:n_train + n_val]
    test_data = all_images[n_train + n_val:]

    print(f"   Train: {len(train_data)} obrazów")
    print(f"   Val: {len(val_data)} obrazów")
    print(f"   Test: {len(test_data)} obrazów")

    # Zapisz w formacie YOLO
    print("\n5. Zapisywanie w formacie YOLO...")

    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_images = [item[0] for item in split_data]
        split_anns = [ann for item in split_data for ann in item[1]]
        save_yolo_format(split_images, split_anns, OUTPUT_BASE_DIR / split_name, coco_data)
        print(f"   Zapisano {split_name}: {len(split_images)} obrazów")

    # Stwórz plik data.yaml dla YOLO
    yaml_content = f"""# Red Ball Dataset
path: {OUTPUT_BASE_DIR.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: ball

# Number of classes
nc: 1
"""
    yaml_path = OUTPUT_BASE_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"   Zapisano konfigurację: {yaml_path}")

    # Usuń katalog tymczasowy
    temp_dir = OUTPUT_BASE_DIR / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("PODSUMOWANIE:")
    print(f"  Oryginalne obrazy: {len(coco_data['images'])}")
    print(f"  Wszystkie obrazy (z augmentacjami): {n_total}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"  Katalog wyjściowy: {OUTPUT_BASE_DIR}")
    print(f"  Plik konfiguracyjny: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
