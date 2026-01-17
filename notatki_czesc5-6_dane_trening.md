# Rozszerzone notatki: Przygotowanie danych i Trening modelu YOLO

## Spis treści
1. [Część 5: Przygotowanie danych](#część-5-przygotowanie-danych)
   - [Formaty adnotacji](#formaty-adnotacji)
   - [Struktura projektu](#struktura-projektu)
   - [Narzędzia do adnotacji](#narzędzia-do-adnotacji)
   - [Data Augmentation](#data-augmentation)
   - [Balansowanie klas](#balansowanie-klas)
2. [Część 6: Trening modelu](#część-6-trening-modelu)
   - [Transfer Learning](#transfer-learning)
   - [Hiperparametry](#hiperparametry)
   - [Learning Rate Schedule](#learning-rate-schedule)
   - [Wymagania sprzętowe](#wymagania-sprzętowe)
   - [Monitoring i metryki](#monitoring-i-metryki)
   - [Overfitting i regularyzacja](#overfitting-i-regularyzacja)

---

# Część 5: Przygotowanie danych

## Wprowadzenie

"Garbage in, garbage out" - ta zasada jest szczególnie prawdziwa w uczeniu maszynowym. Jakość danych treningowych bezpośrednio przekłada się na jakość modelu. Nawet najlepsza architektura sieci neuronowej nie pomoże, jeśli dane są źle przygotowane, niepoprawnie zanotowane lub niereprezentacyjne.

### Kluczowe aspekty przygotowania danych:
1. **Jakość adnotacji** - precyzyjne bounding boxy
2. **Reprezentatywność** - dane podobne do rzeczywistego zastosowania
3. **Różnorodność** - różne warunki oświetlenia, kąty, rozmiary
4. **Balans** - proporcjonalna reprezentacja wszystkich klas
5. **Ilość** - wystarczająca liczba przykładów na klasę

---

## Formaty adnotacji

### Format YOLO (.txt)

Format natywny dla Ultralytics YOLO. Jeden plik tekstowy na jeden obraz.

**Struktura linii:**
```
<class_id> <x_center> <y_center> <width> <height>
```

**Charakterystyka:**
- Wszystkie wartości znormalizowane do zakresu [0.0, 1.0]
- Współrzędne względem rozmiaru obrazu
- Środek bounding boxa (nie róg!)
- `class_id` zaczyna się od 0

**Przykład pliku `image001.txt`:**
```
0 0.45 0.52 0.30 0.40
1 0.78 0.35 0.15 0.25
0 0.20 0.60 0.18 0.30
```

**Konwersja z pikseli do formatu YOLO:**

```python
def convert_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Konwertuje współrzędne pikseli (x1,y1,x2,y2) do formatu YOLO.

    Args:
        x1, y1: lewy górny róg
        x2, y2: prawy dolny róg
        img_width, img_height: rozmiar obrazu w pikselach

    Returns:
        x_center, y_center, width, height: znormalizowane wartości
    """
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return x_center, y_center, width, height

# Przykład
# Obraz 1920x1080, bbox od (400,200) do (700,600)
x_c, y_c, w, h = convert_to_yolo(400, 200, 700, 600, 1920, 1080)
print(f"{x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}")
# Output: 0.2865 0.3704 0.1563 0.3704
```

**Konwersja z formatu YOLO do pikseli:**

```python
def convert_from_yolo(x_center, y_center, width, height, img_width, img_height):
    """
    Konwertuje format YOLO do współrzędnych pikseli.
    """
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)

    return x1, y1, x2, y2
```

---

### Format COCO (JSON)

Standard przemysłowy używany w wielu benchmarkach i konkurencjach.

**Struktura pliku:**
```json
{
    "info": {
        "description": "My Dataset",
        "version": "1.0",
        "year": 2024
    },
    "licenses": [],
    "images": [
        {
            "id": 1,
            "file_name": "image001.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [400, 200, 300, 400],
            "area": 120000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 0, "name": "cat", "supercategory": "animal"},
        {"id": 1, "name": "dog", "supercategory": "animal"}
    ]
}
```

**Format bbox w COCO:**
```
[x_min, y_min, width, height]  # w pikselach
```

**Konwersja COCO → YOLO:**

```python
import json
import os

def coco_to_yolo(coco_json_path, output_dir, images_dir):
    """
    Konwertuje adnotacje COCO JSON do formatu YOLO.
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Mapowanie image_id -> image info
    images = {img['id']: img for img in coco['images']}

    # Grupowanie adnotacji po image_id
    from collections import defaultdict
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    for image_id, anns in annotations_by_image.items():
        img_info = images[image_id]
        img_w, img_h = img_info['width'], img_info['height']

        # Nazwa pliku .txt
        txt_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            for ann in anns:
                # COCO bbox: [x, y, width, height]
                x, y, w, h = ann['bbox']

                # Konwersja do YOLO
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                class_id = ann['category_id']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Użycie
coco_to_yolo('annotations.json', 'labels/', 'images/')
```

---

### Format Pascal VOC (XML)

Starszy format, wciąż spotykany w niektórych datasetach.

**Struktura pliku XML:**
```xml
<annotation>
    <folder>images</folder>
    <filename>image001.jpg</filename>
    <size>
        <width>1920</width>
        <height>1080</height>
        <depth>3</depth>
    </size>
    <object>
        <name>cat</name>
        <bndbox>
            <xmin>400</xmin>
            <ymin>200</ymin>
            <xmax>700</xmax>
            <ymax>600</ymax>
        </bndbox>
    </object>
    <object>
        <name>dog</name>
        <bndbox>
            <xmin>800</xmin>
            <ymin>300</ymin>
            <xmax>1000</xmax>
            <ymax>500</ymax>
        </bndbox>
    </object>
</annotation>
```

**Konwersja VOC → YOLO:**

```python
import xml.etree.ElementTree as ET
import os

def voc_to_yolo(xml_path, class_names):
    """
    Konwertuje pojedynczy plik Pascal VOC XML do formatu YOLO.

    Args:
        xml_path: ścieżka do pliku XML
        class_names: lista nazw klas ['cat', 'dog', ...]

    Returns:
        lista linii w formacie YOLO
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_names:
            continue

        class_id = class_names.index(class_name)

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Konwersja do YOLO
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines

# Konwersja całego folderu
def convert_voc_folder(xml_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        lines = voc_to_yolo(xml_path, class_names)

        txt_name = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))
```

---

### Porównanie formatów

| Cecha | YOLO | COCO | Pascal VOC |
|-------|------|------|------------|
| Format pliku | .txt | .json | .xml |
| Ilość plików | 1 per obraz | 1 dla całego datasetu | 1 per obraz |
| Współrzędne | znormalizowane | piksele | piksele |
| Bbox format | center + size | corner + size | two corners |
| Segmentacja | osobne pliki | polygons w JSON | ograniczone |
| Prostota | ★★★★★ | ★★★ | ★★★ |

**Rekomendacja:** Używaj formatu YOLO z Ultralytics - najprostszy workflow.

---

## Struktura projektu

### Standardowa struktura folderów

```
my_dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Plik data.yaml

```yaml
# Ścieżki do danych (absolutne lub względne)
path: /home/user/my_dataset
train: train/images
val: val/images
test: test/images  # opcjonalne

# Liczba klas
nc: 3

# Nazwy klas (kolejność ważna - odpowiada class_id)
names:
  0: cat
  1: dog
  2: bird

# Lub jako lista
# names: ['cat', 'dog', 'bird']
```

### Proporcje podziału danych

| Zbiór | Proporcja | Przeznaczenie |
|-------|-----------|---------------|
| Train | 70-80% | Uczenie wag modelu |
| Val | 10-20% | Walidacja podczas treningu, dobór hiperparametrów |
| Test | 10-20% | Finalna ocena (nie używany podczas treningu) |

**Skrypt do podziału danych:**

```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images, source_labels, output_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Dzieli dataset na train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    # Lista wszystkich obrazów
    images = [f for f in os.listdir(source_images)
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        img_dir = Path(output_dir) / split_name / 'images'
        lbl_dir = Path(output_dir) / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_name in split_images:
            # Kopiuj obraz
            src_img = Path(source_images) / img_name
            dst_img = img_dir / img_name
            shutil.copy2(src_img, dst_img)

            # Kopiuj label
            lbl_name = Path(img_name).stem + '.txt'
            src_lbl = Path(source_labels) / lbl_name
            if src_lbl.exists():
                dst_lbl = lbl_dir / lbl_name
                shutil.copy2(src_lbl, dst_lbl)

        print(f"{split_name}: {len(split_images)} images")

# Użycie
split_dataset('all_images/', 'all_labels/', 'dataset/')
```

---

## Narzędzia do adnotacji

### LabelImg (Open-source)

**Instalacja:**
```bash
pip install labelImg
```

**Uruchomienie:**
```bash
labelImg
# lub z parametrami
labelImg /path/to/images /path/to/predefined_classes.txt
```

**Cechy:**
- Prosty, lekki interfejs
- Wsparcie dla YOLO, Pascal VOC, COCO
- Tylko bounding boxes (bez segmentacji)
- Skróty klawiszowe: W (nowy box), A/D (poprzedni/następny obraz)

**Plik klas (predefined_classes.txt):**
```
cat
dog
bird
```

---

### CVAT (Open-source)

**Instalacja (Docker):**
```bash
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d
# Dostęp: http://localhost:8080
```

**Cechy:**
- Webowy interfejs
- Bbox, polygons, polylines, points
- Tracking obiektów w wideo
- Współpraca zespołowa
- Automatyczne adnotacje z AI
- Eksport do wielu formatów

**Eksport do YOLO:**
1. Otwórz projekt → Tasks
2. Actions → Export task dataset
3. Format: YOLO 1.1
4. Pobierz ZIP

---

### Roboflow (Freemium)

**Cechy:**
- Platforma chmurowa
- Automatyczne augmentacje
- Preprocessing (resize, auto-orient)
- Health check datasetu
- One-click eksport do Ultralytics
- Wersjonowanie datasetów

**Integracja z Ultralytics:**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("project-name")
dataset = project.version(1).download("yolov8")

# Zwraca ścieżkę do data.yaml
print(dataset.location)
```

---

### Label Studio (Open-source)

**Instalacja:**
```bash
pip install label-studio
label-studio start
# Dostęp: http://localhost:8080
```

**Cechy:**
- Multi-task (CV, NLP, Audio)
- ML-assisted labeling (pre-annotations)
- Konfigurowalny interfejs
- REST API
- Enterprise version z więcej funkcji

**Konfiguracja dla detekcji obiektów:**
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="cat" background="blue"/>
    <Label value="dog" background="green"/>
  </RectangleLabels>
</View>
```

---

## Data Augmentation

### Dlaczego augmentacje?

1. **Zwiększenie rozmiaru datasetu** - więcej przykładów treningowych
2. **Regularyzacja** - zapobieganie overfittingowi
3. **Robustność** - model radzi sobie z różnymi warunkami
4. **Wyrównanie klas** - więcej przykładów rzadkich klas

### Podstawowe transformacje

```python
import albumentations as A

# Transformacje geometryczne
geometric_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),  # tylko dla niektórych domen!
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.5
    ),
    A.RandomCrop(width=512, height=512, p=0.3),
], bbox_params=A.BboxParams(format='yolo'))

# Transformacje kolorystyczne
color_transforms = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ToGray(p=0.1),
])
```

### Mosaic Augmentation

Łączy 4 obrazy w jeden, co:
- Zwiększa kontekst
- Pokazuje obiekty w różnych skalach
- Redukuje wymagania batch size

```python
def mosaic_augmentation(images, labels, img_size=640):
    """
    Tworzy obraz mozaikowy z 4 obrazów.

    Args:
        images: lista 4 obrazów (numpy arrays)
        labels: lista 4 list bboxów w formacie YOLO
        img_size: rozmiar wyjściowego obrazu

    Returns:
        mosaic_img, mosaic_labels
    """
    import numpy as np
    import cv2

    s = img_size
    # Punkt podziału (losowy)
    xc = int(np.random.uniform(s * 0.25, s * 0.75))
    yc = int(np.random.uniform(s * 0.25, s * 0.75))

    mosaic_img = np.zeros((s, s, 3), dtype=np.uint8)
    mosaic_labels = []

    # Pozycje dla 4 obrazów
    placements = [
        (0, 0, xc, yc),           # top-left
        (xc, 0, s, yc),           # top-right
        (0, yc, xc, s),           # bottom-left
        (xc, yc, s, s)            # bottom-right
    ]

    for i, (img, lbls) in enumerate(zip(images, labels)):
        x1, y1, x2, y2 = placements[i]
        h, w = y2 - y1, x2 - x1

        # Resize obrazu do wymaganego rozmiaru
        img_resized = cv2.resize(img, (w, h))
        mosaic_img[y1:y2, x1:x2] = img_resized

        # Przekształcenie bboxów
        for lbl in lbls:
            cls_id, xc_n, yc_n, w_n, h_n = lbl

            # Denormalizacja względem oryginalnego obrazu
            orig_h, orig_w = img.shape[:2]
            xc_px = xc_n * orig_w
            yc_px = yc_n * orig_h
            w_px = w_n * orig_w
            h_px = h_n * orig_h

            # Skalowanie do rozmiaru kafelka
            scale_x = w / orig_w
            scale_y = h / orig_h
            xc_px = xc_px * scale_x + x1
            yc_px = yc_px * scale_y + y1
            w_px = w_px * scale_x
            h_px = h_px * scale_y

            # Normalizacja względem mozaiki
            xc_n_new = xc_px / s
            yc_n_new = yc_px / s
            w_n_new = w_px / s
            h_n_new = h_px / s

            mosaic_labels.append([cls_id, xc_n_new, yc_n_new, w_n_new, h_n_new])

    return mosaic_img, mosaic_labels
```

### MixUp Augmentation

Mieszanie dwóch obrazów i ich etykiet:

```python
def mixup_augmentation(img1, labels1, img2, labels2, alpha=0.5):
    """
    MixUp: łączy dwa obrazy z wagą alpha.
    """
    import numpy as np

    # Lambda z rozkładu Beta
    lam = np.random.beta(alpha, alpha)

    # Mieszanie obrazów
    mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)

    # Dla detekcji: zachowujemy wszystkie boxy z obu obrazów
    # (można też ważyć confidence scores przez lam)
    mixed_labels = labels1 + labels2

    return mixed_img, mixed_labels
```

### Augmentacje w Ultralytics

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Domyślne augmentacje podczas treningu
model.train(
    data='data.yaml',
    epochs=100,

    # Augmentacje kolorystyczne
    hsv_h=0.015,      # Hue (+/- 1.5%)
    hsv_s=0.7,        # Saturation (+/- 70%)
    hsv_v=0.4,        # Value/Brightness (+/- 40%)

    # Augmentacje geometryczne
    degrees=0.0,      # Rotacja (+/- stopnie)
    translate=0.1,    # Przesunięcie (+/- 10%)
    scale=0.5,        # Skalowanie (+/- 50%)
    shear=0.0,        # Shear (+/- stopnie)
    perspective=0.0,  # Perspektywa

    # Flip
    flipud=0.0,       # Vertical flip (prawdopodobieństwo)
    fliplr=0.5,       # Horizontal flip

    # Zaawansowane
    mosaic=1.0,       # Mosaic (prawdopodobieństwo)
    mixup=0.0,        # MixUp (0.0-1.0)
    copy_paste=0.0,   # Copy-paste dla segmentacji

    # Close mosaic w ostatnich epokach
    close_mosaic=10,  # Wyłącz mosaic na ostatnie 10 epok
)
```

---

## Balansowanie klas

### Problem niezbalansowanego datasetu

Typowe proporcje w rzeczywistych datasetach:
- Klasa "samochód": 5000 przykładów
- Klasa "motocykl": 500 przykładów
- Klasa "rower": 100 przykładów

Model będzie faworyzował częste klasy!

### Rozwiązania

#### 1. Oversampling

```python
import os
import shutil
from collections import Counter

def oversample_rare_classes(labels_dir, images_dir, target_count=1000):
    """
    Duplikuje obrazy rzadkich klas.
    """
    # Zlicz instancje każdej klasy
    class_counts = Counter()
    file_classes = {}

    for lbl_file in os.listdir(labels_dir):
        if not lbl_file.endswith('.txt'):
            continue

        with open(os.path.join(labels_dir, lbl_file)) as f:
            classes_in_file = set()
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
                classes_in_file.add(cls_id)
            file_classes[lbl_file] = classes_in_file

    # Znajdź obrazy z rzadkimi klasami i duplikuj
    for lbl_file, classes in file_classes.items():
        for cls_id in classes:
            if class_counts[cls_id] < target_count:
                copies_needed = target_count // class_counts[cls_id]

                for i in range(copies_needed):
                    # Nowa nazwa pliku
                    base_name = lbl_file.replace('.txt', '')
                    new_name = f"{base_name}_dup{i}"

                    # Kopiuj label
                    shutil.copy(
                        os.path.join(labels_dir, lbl_file),
                        os.path.join(labels_dir, f"{new_name}.txt")
                    )

                    # Kopiuj obraz
                    for ext in ['.jpg', '.jpeg', '.png']:
                        src_img = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(src_img):
                            shutil.copy(src_img,
                                os.path.join(images_dir, new_name + ext))
                            break
```

#### 2. Class Weights

W Ultralytics można użyć parametru `cls` w loss function:

```python
# Wyższa waga dla rzadkich klas
# (wymaga modyfikacji kodu lub użycia custom loss)
```

#### 3. Focal Loss

Automatycznie zwiększa wagę trudnych (rzadkich) przykładów:

$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

gdzie $\gamma = 2$ skupia trening na trudnych przykładach.

### Analiza datasetu

```python
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(data_yaml):
    """
    Analizuje rozkład klas w datasecie.
    """
    import yaml

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    labels_dir = os.path.join(data['path'], data['train'].replace('images', 'labels'))

    class_counts = Counter()

    for lbl_file in os.listdir(labels_dir):
        if not lbl_file.endswith('.txt'):
            continue

        with open(os.path.join(labels_dir, lbl_file)) as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    # Wykres
    names = data['names']
    classes = [names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel('Klasa')
    plt.ylabel('Liczba instancji')
    plt.title('Rozkład klas w datasecie')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

    return class_counts

# Użycie
counts = analyze_dataset('data.yaml')
```

---

# Część 6: Trening modelu

## Transfer Learning

### Czym jest Transfer Learning?

Transfer learning polega na wykorzystaniu modelu wytrenowanego na dużym datasecie (np. COCO, ImageNet) jako punktu startowego dla nowego zadania.

**Korzyści:**
1. **Szybsza konwergencja** - model już "wie" jak rozpoznawać podstawowe cechy
2. **Lepsza dokładność** - szczególnie przy małych datasetach
3. **Mniej danych potrzebnych** - cechy niskiego poziomu są uniwersalne
4. **Stabilniejszy trening** - dobre wartości początkowe wag

### Pretrenowane wagi COCO

Dataset COCO (Common Objects in Context):
- 80 klas obiektów
- 330,000 obrazów
- 1.5 miliona instancji obiektów

```python
from ultralytics import YOLO

# Ładowanie z pretrenowanymi wagami COCO
model = YOLO('yolo11n.pt')  # lub yolo11s.pt, yolo11m.pt, etc.

# Trening na własnych danych
model.train(data='my_data.yaml', epochs=100)
```

### Strategie fine-tuningu

#### 1. Pełny fine-tuning (zalecane)
```python
model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=100)
```
- Trenuje wszystkie warstwy
- Najlepsza dokładność
- Wymaga więcej danych

#### 2. Freeze backbone
```python
model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=100, freeze=10)
```
- Zamraża pierwsze N warstw
- Szybszy trening
- Mniej podatne na overfitting
- Dobre dla małych datasetów

#### 3. Trening od zera (rzadko używane)
```python
model = YOLO('yolo11n.yaml')  # Tylko konfiguracja, bez wag
model.train(data='data.yaml', epochs=300)
```
- Wymaga bardzo dużego datasetu (>10,000 obrazów)
- Długi czas treningu
- Tylko gdy domain jest bardzo różny od COCO

### Kiedy używać której strategii?

| Rozmiar datasetu | Podobieństwo do COCO | Strategia |
|------------------|---------------------|-----------|
| <500 obrazów | Wysokie | Freeze + mało epok |
| 500-5000 | Wysokie | Pełny fine-tuning |
| 500-5000 | Niskie | Freeze najpierw, potem fine-tuning |
| >5000 | Dowolne | Pełny fine-tuning |
| >50000 | Bardzo niskie | Rozważ trening od zera |

---

## Hiperparametry

### Learning Rate

Najważniejszy hiperparametr!

```python
model.train(
    lr0=0.01,    # Początkowy learning rate
    lrf=0.01,    # Końcowy LR jako % początkowego (0.01 = 0.0001 końcowe)
)
```

**Jak dobrać LR:**
- Za duży → niestabilny trening, loss skacze
- Za mały → wolna konwergencja
- Typowe wartości: 0.001 - 0.01 dla fine-tuningu

**Learning Rate Finder:**
```python
# Nieformalny sposób - obserwuj training loss
# Jeśli loss rośnie → zmniejsz LR
# Jeśli loss maleje bardzo wolno → zwiększ LR
```

### Batch Size

```python
model.train(
    batch=16,    # Stały batch size
    batch=-1,    # Auto-batch (dopasowuje do pamięci GPU)
)
```

**Wpływ batch size:**
- Większy → stabilniejsze gradienty, lepsza generalizacja
- Mniejszy → szybsze iteracje, więcej szumu
- Ograniczony przez pamięć GPU

**Reguła kciuka:**
- Zwiększ LR proporcjonalnie do batch size
- `batch=32, lr0=0.01` → `batch=64, lr0=0.02`

### Epochs

```python
model.train(
    epochs=100,     # Liczba epok
    patience=50,    # Early stopping po N epokach bez poprawy
)
```

**Ile epok?**
- Typowo: 100-300 dla fine-tuningu
- Obserwuj validation mAP
- Użyj early stopping

### Image Size

```python
model.train(
    imgsz=640,    # Rozmiar obrazu (musi być wielokrotnością 32)
)
```

**Wpływ rozmiaru:**
- Większy → lepsza dokładność (szczególnie małe obiekty)
- Większy → więcej pamięci, wolniejszy trening
- Typowe wartości: 320, 416, 640, 1280

---

## Learning Rate Schedule

### Warm-up

Stopniowe zwiększanie LR na początku treningu:

```python
model.train(
    warmup_epochs=3.0,       # Liczba epok warm-up
    warmup_momentum=0.8,     # Momentum podczas warm-up
    warmup_bias_lr=0.1,      # LR dla bias podczas warm-up
)
```

**Dlaczego warm-up?**
- Gradienty na początku mogą być duże i niestabilne
- Stopniowe zwiększanie LR stabilizuje trening
- Szczególnie ważne przy dużych batch size

### Cosine Annealing

Po warm-up, LR maleje zgodnie z funkcją cosinus:

$$LR(t) = LR_{final} + \frac{1}{2}(LR_{start} - LR_{final})(1 + \cos(\pi \cdot \frac{t}{T}))$$

```
LR
│
│   ****
│  *    ***
│ *        ***
│*            ***
│                 ****
└─────────────────────── epoch
  warm-up → cosine decay
```

**Korzyści:**
- Płynne zmniejszanie LR
- Lepsze minima lokalne
- Lepsza generalizacja

---

## Wymagania sprzętowe

### GPU

| Model | Min VRAM | Batch przy 8GB | Batch przy 16GB |
|-------|----------|----------------|-----------------|
| YOLO11n | 4 GB | 32-64 | 64-128 |
| YOLO11s | 4 GB | 16-32 | 32-64 |
| YOLO11m | 6 GB | 8-16 | 16-32 |
| YOLO11l | 8 GB | 4-8 | 8-16 |
| YOLO11x | 12 GB | 2-4 | 4-8 |

**Sprawdzanie dostępnej pamięci:**
```bash
nvidia-smi
```

### CPU Training

```python
model.train(data='data.yaml', device='cpu')
```

- 10-100x wolniejsze niż GPU
- OK dla małych datasetów i prototypowania
- Bez ograniczeń pamięci (poza RAM)

### Multi-GPU

```python
model.train(data='data.yaml', device='0,1,2,3')
```

- DataParallel automatycznie
- Batch size = batch_per_gpu * num_gpus
- Wymaga odpowiedniego skalowania LR

### Google Colab

```python
# Sprawdź przydzielony GPU
!nvidia-smi

# Zamontuj Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Zainstaluj Ultralytics
!pip install ultralytics

# Trenuj
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(data='/content/drive/MyDrive/dataset/data.yaml', epochs=100)
```

**Limity Colab Free:**
- T4 GPU (16GB VRAM)
- ~12h limit sesji
- Przerwy w dostępie GPU

---

## Monitoring i metryki

### Metryki Loss

```
box_loss   - błąd lokalizacji bounding boxów (CIoU)
cls_loss   - błąd klasyfikacji (Binary Cross-Entropy)
dfl_loss   - Distribution Focal Loss (precyzja granic)
```

**Jak interpretować:**
- Loss powinien maleć przez epoki
- Różnica train/val loss wskazuje na overfitting
- Nagły skok loss = problemy (LR za duży, złe dane)

### Metryki Detekcji

**Precision:**
$$Precision = \frac{TP}{TP + FP}$$
- "Ile z moich detekcji jest poprawnych?"

**Recall:**
$$Recall = \frac{TP}{TP + FN}$$
- "Ile z istniejących obiektów wykryłem?"

**mAP (mean Average Precision):**
- mAP@0.5 - IoU threshold = 0.5 (standardowy)
- mAP@0.5:0.95 - średnia z IoU od 0.5 do 0.95 (rygorystyczny)

### Confusion Matrix

```python
# Po treningu
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val()

# Confusion matrix jest automatycznie generowana w:
# runs/detect/train/confusion_matrix.png
```

**Interpretacja:**
- Przekątna = poprawne klasyfikacje
- Poza przekątną = pomyłki
- Ostatni wiersz/kolumna = background (FN/FP)

### Narzędzia wizualizacji

#### TensorBoard

```python
model.train(data='data.yaml', project='runs/train')

# W osobnym terminalu
tensorboard --logdir runs/train
# Otwórz http://localhost:6006
```

#### Weights & Biases

```python
# Instalacja
pip install wandb

# Logowanie
import wandb
wandb.login()

# Ultralytics automatycznie loguje do W&B jeśli zainstalowany
model.train(data='data.yaml', epochs=100)
```

---

## Overfitting i regularyzacja

### Objawy overfittingu

1. Train loss maleje, val loss rośnie
2. Train mAP bardzo wysokie, val mAP znacznie niższe
3. Model "pamięta" dane treningowe zamiast generalizować

```
Loss
│
│    val loss
│       ****
│      *
│     *
│    *
│   *     train loss
│  ********************************
└────────────────────────────────── epoch
            ↑
     overfitting zaczyna się tutaj
```

### Metody zapobiegania

#### 1. Więcej danych
Najlepsza metoda! Zbierz więcej zróżnicowanych danych.

#### 2. Data Augmentation
```python
model.train(
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
)
```

#### 3. Regularyzacja

```python
model.train(
    weight_decay=0.0005,  # L2 regularization
    dropout=0.0,          # Dropout w head (0.0-0.5)
)
```

#### 4. Early Stopping

```python
model.train(
    patience=50,  # Zatrzymaj po 50 epokach bez poprawy val mAP
)
```

#### 5. Mniejszy model

```python
# Zamiast yolo11x użyj mniejszego wariantu
model = YOLO('yolo11n.pt')  # nano
model = YOLO('yolo11s.pt')  # small
```

### Underfitting

Gdy model nie uczy się wystarczająco:
- Train i val loss wysokie
- mAP niskie

**Rozwiązania:**
- Większy model
- Więcej epok
- Większy learning rate
- Mniej augmentacji (model może być zdezorientowany)

---

## Kompletny przykład treningu

```python
from ultralytics import YOLO

# 1. Załaduj model z pretrenowanymi wagami
model = YOLO('yolo11n.pt')

# 2. Trenuj na własnych danych
results = model.train(
    # Podstawowe
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,

    # Urządzenie
    device=0,  # GPU 0

    # Learning rate
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,

    # Augmentacje
    mosaic=1.0,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    # Regularyzacja
    weight_decay=0.0005,
    dropout=0.0,

    # Early stopping
    patience=50,

    # Zapisywanie
    save=True,
    save_period=10,  # Zapisuj co 10 epok
    project='runs/my_project',
    name='experiment_1',
    exist_ok=True,

    # Walidacja
    val=True,

    # Verbose
    verbose=True,
    plots=True,
)

# 3. Ewaluacja na zbiorze testowym
metrics = model.val(data='path/to/data.yaml', split='test')
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

# 4. Eksport modelu
model.export(format='onnx')  # lub 'torchscript', 'tflite', etc.
```

### Struktura wyników

```
runs/my_project/experiment_1/
├── weights/
│   ├── best.pt          # Najlepszy model (wg val mAP)
│   └── last.pt          # Model z ostatniej epoki
├── args.yaml            # Użyte hiperparametry
├── results.csv          # Metryki per epoka
├── results.png          # Wykresy metryk
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
├── PR_curve.png         # Precision-Recall curve
└── train_batch*.jpg     # Przykładowe batche treningowe
```

---

## Checklist przed treningiem

- [ ] Dane przygotowane i sprawdzone
  - [ ] Poprawna struktura folderów
  - [ ] Pliki labels odpowiadają images
  - [ ] Format YOLO (znormalizowane współrzędne)

- [ ] data.yaml poprawny
  - [ ] Ścieżki absolutne lub względne
  - [ ] Poprawna liczba klas (nc)
  - [ ] Nazwy klas w dobrej kolejności

- [ ] GPU dostępne
  - [ ] `nvidia-smi` pokazuje dostępną pamięć
  - [ ] Sterowniki CUDA zainstalowane

- [ ] Wybrany odpowiedni model
  - [ ] Rozmiar dopasowany do sprzętu
  - [ ] Balans accuracy/speed

---

## Typowe problemy i rozwiązania

| Problem | Możliwa przyczyna | Rozwiązanie |
|---------|-------------------|-------------|
| Loss = NaN | LR za duży | Zmniejsz lr0 |
| Val loss rośnie | Overfitting | Więcej augmentacji, early stopping |
| mAP = 0 | Błędne dane | Sprawdź format labels |
| OOM (out of memory) | Za duży batch | Zmniejsz batch, użyj mniejszego modelu |
| Wolny trening | Brak GPU | Sprawdź `device`, reinstaluj CUDA |
| Niska dokładność | Za mało danych | Więcej danych, transfer learning |

---

## Podsumowanie

### Przygotowanie danych
1. Wybierz format (YOLO zalecane)
2. Zorganizuj strukturę folderów
3. Stwórz data.yaml
4. Sprawdź jakość adnotacji
5. Zastosuj augmentacje
6. Zbilansuj klasy jeśli potrzeba

### Trening
1. Użyj pretrenowanych wag
2. Dostosuj hiperparametry
3. Monitoruj metryki
4. Użyj early stopping
5. Analizuj wyniki (confusion matrix)
6. Iteruj i ulepszaj

---

*Notatki przygotowane do warsztatu YOLO - Część 5-6*
