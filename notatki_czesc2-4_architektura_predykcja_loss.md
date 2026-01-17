# Notatki rozszerzone: Części 2-4 - Architektura, Predykcja, Loss Function

## Spis treści
1. [Część 2: Architektura sieci](#część-2-architektura-sieci)
2. [Część 3: Jak działa predykcja](#część-3-jak-działa-predykcja)
3. [Część 4: Funkcja straty (Loss Function)](#część-4-funkcja-straty-loss-function)

---

# Część 2: Architektura sieci

## 2.1 Wprowadzenie do sieci neuronowych (przypomnienie)

### Podstawowe pojęcia

**Neuron:**
- Podstawowa jednostka obliczeniowa
- Oblicza sumę ważoną wejść + bias
- Przepuszcza przez funkcję aktywacji
```
output = activation(sum(weights * inputs) + bias)
```

**Wagi (weights):**
- Parametry sieci uczące się podczas treningu
- Określają siłę połączeń między neuronami
- Inicjalizowane losowo, optymalizowane przez backpropagation

**Funkcje aktywacji:**
| Funkcja | Wzór | Zastosowanie |
|---------|------|--------------|
| ReLU | max(0, x) | Najpopularniejsza, szybka |
| Leaky ReLU | max(0.01x, x) | Zapobiega "dying ReLU" |
| Mish | x * tanh(softplus(x)) | YOLOv4+, gładka |
| SiLU/Swish | x * sigmoid(x) | YOLOv5+, self-gated |
| Sigmoid | 1/(1+e^(-x)) | Output (0-1), prawdopodobieństwa |

### Warstwy konwolucyjne (CNN)

**Convolution:**
- Filtr (kernel) przesuwa się po obrazie
- Wykrywa wzorce (krawędzie, tekstury, kształty)
- Parametry: kernel_size, stride, padding, filters

```
Przykład: Conv2D(64, kernel_size=3, stride=1, padding=1)
- 64 filtrów 3×3
- Stride 1 = przesuń o 1 piksel
- Padding 1 = zachowaj rozdzielczość
```

**Pooling:**
- Redukcja rozdzielczości przestrzennej
- Max Pooling: bierze maksymalną wartość
- Average Pooling: bierze średnią
- Stride Pooling (Conv ze stride>1): nowoczesna alternatywa

**Batch Normalization:**
- Normalizuje aktywacje w mini-batch
- Stabilizuje trening, pozwala na większy learning rate
- Formuła: `y = gamma * (x - mean) / sqrt(var + eps) + beta`

**Skip/Residual Connections:**
- Dodaje input bezpośrednio do output bloku
- Umożliwia trening bardzo głębokich sieci
- Zapobiega vanishing gradients
- `output = F(x) + x` (ResNet)

### Interaktywna wizualizacja

**TensorFlow Playground:** https://playground.tensorflow.org/

Świetne narzędzie do zrozumienia:
- Jak sieci uczą się granic decyzyjnych
- Wpływ liczby warstw i neuronów
- Różne funkcje aktywacji
- Overfitting vs underfitting

---

## 2.2 Architektura YOLO - przegląd

### Trzy główne komponenty

```
Obraz wejściowy
      ↓
┌─────────────────┐
│    BACKBONE     │  ← Ekstrakcja cech
│  (CSPDarknet)   │
└────────┬────────┘
         ↓
┌─────────────────┐
│      NECK       │  ← Agregacja multi-scale
│   (FPN + PANet) │
└────────┬────────┘
         ↓
┌─────────────────┐
│      HEAD       │  ← Predykcja bbox + klasy
│  (Detect Layer) │
└────────┬────────┘
         ↓
   Detekcje
```

---

## 2.3 Backbone - Ekstraktor cech

### Zadanie
Przekształcenie obrazu wejściowego (np. 640×640×3) na hierarchię map cech (feature maps) o rosnącej semantyce i malejącej rozdzielczości.

### Hierarchia cech

| Poziom | Rozdzielczość | Co wykrywa |
|--------|---------------|------------|
| Wczesne warstwy | Wysoka (320×320) | Krawędzie, tekstury, gradienty |
| Środkowe warstwy | Średnia (80×80) | Części obiektów (oczy, koła, okna) |
| Późne warstwy | Niska (20×20) | Semantyka wysokiego poziomu (kot, samochód) |

### Darknet-53 (YOLOv3)

**Architektura:**
- 53 warstwy konwolucyjne
- Residual blocks (inspiracja ResNet)
- Brak fully connected layers

**Blok residualny Darknet:**
```
input
  ├── Conv 1×1 (reduce channels)
  ├── Conv 3×3 (extract features)
  └── + input (skip connection)
```

**Zalety:**
- Szybszy niż ResNet-152 przy podobnej dokładności
- Lepszy dla detekcji niż czyste sieci klasyfikacyjne

### CSPDarknet (YOLOv4, v5)

**CSP = Cross-Stage Partial Networks**

**Problem z głębokimi sieciami:**
- Dużo redundantnych gradientów
- Wysokie koszty obliczeniowe
- Powolna konwergencja

**Rozwiązanie CSP:**
1. Dzieli feature map na dwie części (split)
2. Część 1 → przechodzi przez gęste bloki konwolucyjne
3. Część 2 → shortcut (przeskakuje gęste bloki)
4. Łączenie obu części (concatenation)
5. Transition layer

```
      Input
        ↓
    ┌───┴───┐
    │       │
  Part1   Part2
    │       │
 Dense     │
 Block     │
    │       │
    └───┬───┘
        ↓
     Concat
        ↓
   Transition
        ↓
     Output
```

**Korzyści CSP:**
- ~50% redukcja obliczeń przy zachowaniu accuracy
- Lepszy przepływ gradientów
- Mniejsze zużycie pamięci
- Lepsza generalizacja

### C2f Block (YOLOv8)

**Ulepszona wersja CSP dla YOLOv8:**
- "CSP Bottleneck with 2 convolutions"
- Gradient flow przez wiele ścieżek
- Efektywniejsze wykorzystanie feature maps

```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        self.cv1 = Conv(c1, c2, 1)  # 1×1 conv
        self.cv2 = Conv((2 + n) * c2 // 2, c2, 1)  # 1×1 conv
        self.m = nn.ModuleList(
            Bottleneck(c2 // 2, c2 // 2, shortcut) for _ in range(n)
        )
```

### Inne backbone'y

**EfficientRep (YOLOv6):**
- Reparametryzacja dla szybszej inferencji
- Training: wielobranżowa architektura
- Inference: pojedyncza konwolucja (re-param)

**E-ELAN (YOLOv7):**
- Extended Efficient Layer Aggregation Network
- Kontrolowany przepływ gradientów
- Lepsze wykorzystanie parametrów

---

## 2.4 Neck - Agregacja cech

### Problem
- Małe obiekty najlepiej widoczne w wysokorozdzielczych feature maps (wczesne warstwy)
- Duże obiekty wymagają szerokiego receptive field (późne warstwy)
- Jak połączyć najlepsze z obu światów?

### Feature Pyramid Network (FPN)

**Autorzy:** Lin et al., Facebook AI Research, 2017

**Idea:**
1. **Bottom-up pathway:** zwykły backbone (encoder), rozdzielczość maleje
2. **Top-down pathway:** upsampling głębokich cech (2× upsampling)
3. **Lateral connections:** dodawanie cech z bottom-up do top-down (element-wise addition)

```
Bottom-up (Backbone)          Top-down (FPN)
   ↓                              ↑
┌──────┐                     ┌──────┐
│C2 256│─────────────────────→│P2 256│ ← Małe obiekty
└──┬───┘                     └──┬───┘
   ↓                            ↑ (2× upsample + lateral)
┌──────┐                     ┌──────┐
│C3 512│─────────────────────→│P3 256│ ← Średnie obiekty
└──┬───┘                     └──┬───┘
   ↓                            ↑
┌──────┐                     ┌──────┐
│C4 1024│────────────────────→│P4 256│ ← Średnie/duże
└──┬───┘                     └──┬───┘
   ↓                            ↑
┌──────┐                     ┌──────┐
│C5 2048│────────────────────→│P5 256│ ← Duże obiekty
└──────┘                     └──────┘
```

**Lateral connection:**
```python
# P4 = upsample(P5) + Conv1×1(C4)
lateral = self.lateral_conv(c4)  # 1×1 conv, reduce channels
top_down = F.interpolate(p5, scale_factor=2)  # 2× upsample
p4 = lateral + top_down
```

**Wynik FPN:**
- Wszystkie poziomy mają tę samą liczbę kanałów (256)
- Każdy poziom ma bogatą semantykę + odpowiednią rozdzielczość
- Znacząca poprawa dla małych obiektów

### Path Aggregation Network (PANet)

**Autorzy:** Liu et al., 2018

**Rozszerzenie FPN:**
- Dodatkowa ścieżka bottom-up PO FPN
- Cechy z niskich warstw (precyzyjna lokalizacja) szybciej docierają do wyższych poziomów

```
Backbone     FPN (top-down)     PANet (bottom-up)
   ↓              ↑                   ↓
┌──────┐      ┌──────┐           ┌──────┐
│  C3  │─────→│  P3  │───────────→│  N3  │ ← Output
└──┬───┘      └──┬───┘           └──┬───┘
   ↓              ↑                   ↓
┌──────┐      ┌──────┐           ┌──────┐
│  C4  │─────→│  P4  │───────────→│  N4  │ ← Output
└──┬───┘      └──┬───┘           └──┬───┘
   ↓              ↑                   ↓
┌──────┐      ┌──────┐           ┌──────┐
│  C5  │─────→│  P5  │───────────→│  N5  │ ← Output
└──────┘      └──────┘           └──────┘
```

**Dlaczego to pomaga:**
- FPN: semantyka płynie z góry do dołu (top-down)
- PANet: precyzja lokalizacji płynie z dołu do góry (bottom-up)
- Krótsze ścieżki dla informacji z niskich warstw
- Używane w YOLOv4, YOLOv5, YOLOv8

### SPPF (Spatial Pyramid Pooling - Fast)

**Oryginalny SPP (He et al., 2015):**
- Pooling z różnymi rozmiarami kerneli (5×5, 9×9, 13×13)
- Agregacja kontekstu z różnych skal
- Rozszerzenie receptive field bez utraty rozdzielczości

**SPPF (YOLOv5+):**
- Szybsza wersja SPP
- Sekwencyjne 5×5 max pooling (zamiast równoległych dużych kerneli)
- Identyczny wynik, szybsze obliczenia

```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        self.cv1 = Conv(c1, c1 // 2, 1)
        self.cv2 = Conv(c1 * 2, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
```

---

## 2.5 Head - Warstwa detekcyjna

### Zadanie
Przekształcenie feature maps z Neck na finalne predykcje: bounding boxy + klasy + confidence.

### Coupled vs Decoupled Head

**Coupled Head (YOLOv3-v5):**
- Jedna wspólna ścieżka dla klasyfikacji i lokalizacji
- Prostsze, mniej parametrów
- Może być suboptymalne (różne zadania konkurują o te same cechy)

```
Feature Map → Conv → Conv → Output [x, y, w, h, obj, cls1, cls2, ...]
```

**Decoupled Head (YOLOX, YOLOv8):**
- Osobne gałęzie dla różnych zadań
- Klasyfikacja i lokalizacja używają innych cech
- Lepsza konwergencja, wyższa dokładność

```
Feature Map
     ↓
   Conv (shared stem)
     ↓
  ┌──┴──┐
  ↓     ↓
 Cls   Reg
branch branch
  ↓     ↓
Classes  BBox
```

**Korzyści decoupled head:**
- +2-3% mAP w porównaniu do coupled
- Szybsza konwergencja
- Łatwiejsze debugowanie

### Wyjście Head

Dla każdej skali (P3, P4, P5), Head generuje:

| Wyjście | Rozmiar | Opis |
|---------|---------|------|
| Bbox | H × W × 4 | (x, y, w, h) lub (l, t, r, b) |
| Objectness | H × W × 1 | P(object) - czy jest obiekt |
| Classes | H × W × C | P(class_i \| object) |

**Całkowite wyjście (YOLOv8, 640×640 input, 80 klas):**
```
P3 (80×80): 80 × 80 × 85 = 544,000 wartości
P4 (40×40): 40 × 40 × 85 = 136,000 wartości
P5 (20×20): 20 × 20 × 85 = 34,000 wartości
─────────────────────────────────────────────
Razem: 714,000 wartości → 8400 potencjalnych detekcji
```

---

## 2.6 Anchor Boxes vs Anchor-Free

### Anchor-Based (YOLOv2-v5, v7)

**Idea:**
- Predefiniowane "szablony" bounding boxów
- Sieć przewiduje offsety względem anchorów
- Łatwiejsze do nauki (mniejsze wartości do przewidzenia)

**Jak działają anchory:**
```
Dla każdej komórki siatki i każdego anchora:
- Anchor ma wymiary (p_w, p_h)
- Sieć przewiduje: t_x, t_y, t_w, t_h

Dekodowanie:
b_x = sigmoid(t_x) + c_x      # środek x
b_y = sigmoid(t_y) + c_y      # środek y
b_w = p_w * exp(t_w)          # szerokość
b_h = p_h * exp(t_h)          # wysokość
```

**Typowe anchory (YOLOv5):**
```yaml
anchors:
  - [10,13, 16,30, 33,23]      # P3/8 (małe obiekty)
  - [30,61, 62,45, 59,119]     # P4/16 (średnie)
  - [116,90, 156,198, 373,326] # P5/32 (duże)
```

**Auto-anchor (YOLOv5+):**
- K-means clustering na ground truth boxes z datasetu
- Automatyczne dopasowanie anchorów do konkretnego problemu
- `yolo detect train ... --auto-anchor`

**Problemy z anchorami:**
1. Wymagają doboru (ręczny lub k-means)
2. Dużo hiperparametrów (liczba anchorów, rozmiary, proporcje)
3. Problemy z obiektami o nietypowych proporcjach
4. Imbalance: większość anchorów to tło

### Anchor-Free (YOLOX, YOLOv8, YOLO11)

**Idea:**
- Bezpośrednia predykcja współrzędnych
- Brak predefiniowanych szablonów
- Prostszy model, mniej hiperparametrów

**Jak działa (YOLOv8):**
```
Dla każdej komórki (c_x, c_y) na skali ze stride s:
- Sieć przewiduje: l, t, r, b (odległości do krawędzi)

Dekodowanie:
x_min = (c_x - l) * s
y_min = (c_y - t) * s
x_max = (c_x + r) * s
y_max = (c_y + b) * s
```

**Distribution Focal Loss (DFL):**
YOLOv8 używa DFL do przewidywania odległości:
- Zamiast pojedynczej wartości, przewiduje rozkład
- Lepsza lokalizacja granic obiektów
- Bardziej miękka reprezentacja niepewności

**Porównanie:**

| Aspekt | Anchor-based | Anchor-free |
|--------|--------------|-------------|
| Hiperparametry | Dużo (anchory) | Mniej |
| Prędkość | Podobna | Podobna |
| Dokładność | Dobra | Bardzo dobra (nowsze) |
| Łatwość użycia | Wymaga dostrojenia | Plug and play |
| Nietypowe proporcje | Problematyczne | Lepiej |

---

## 2.7 Multi-Scale Predictions

### Dlaczego wiele skal?

**Problem:**
- Małe obiekty (np. 20×20 px): wymagają wysokiej rozdzielczości feature maps
- Duże obiekty (np. 400×400 px): wymagają szerokiego receptive field
- Jedna skala nie pasuje do wszystkich

**Rozwiązanie - 3 skale (YOLOv3+):**

| Skala | Grid (640 input) | Stride | Obiekty | Receptive Field |
|-------|------------------|--------|---------|-----------------|
| P3 | 80×80 | 8 | Małe (8-64 px) | Mały |
| P4 | 40×40 | 16 | Średnie (64-256 px) | Średni |
| P5 | 20×20 | 32 | Duże (256-512 px) | Duży |

**Stride:**
- Co ile pikseli obrazu przypada jedna komórka siatki
- P3 stride 8: każda komórka "widzi" region 8×8 pikseli
- P5 stride 32: każda komórka "widzi" region 32×32 pikseli

### Przypisanie obiektów do skal

**YOLOv3-v5 (anchor-based):**
- Obiekt przypisywany do skali, gdzie jego rozmiar najlepiej pasuje do anchorów
- IoU między ground truth a anchorem > threshold

**YOLOv8 (anchor-free):**
- Task-Aligned Assigner
- Bierze pod uwagę zarówno klasyfikację jak i lokalizację
- Dynamiczne przypisanie podczas treningu

---

# Część 3: Jak działa predykcja

## 3.1 Pipeline predykcji

### Pełny przepływ

```
1. INPUT
   Obraz (dowolny rozmiar)
        ↓
2. PREPROCESSING
   - Resize do 640×640 (lub inna potęga 32)
   - Letterboxing (zachowanie proporcji)
   - Normalizacja (0-1 lub ImageNet)
   - CHW format, batch dimension
        ↓
3. FORWARD PASS
   Backbone → Neck → Head
        ↓
4. RAW OUTPUTS
   Tensor predykcji dla każdej skali
        ↓
5. DECODE
   Raw outputs → współrzędne bbox
        ↓
6. FILTER
   Confidence threshold (usuń słabe)
        ↓
7. NMS
   Usuń duplikaty
        ↓
8. OUTPUT
   Lista detekcji: [(class, bbox, conf), ...]
```

### Preprocessing

**Resize z letterboxing:**
```python
def letterbox(img, new_shape=640):
    # Oblicz skalę zachowując proporcje
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape / shape[0], new_shape / shape[1])

    # Nowy rozmiar
    new_unpad = (int(shape[1] * r), int(shape[0] * r))

    # Padding
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    # Resize i pad
    img = cv2.resize(img, new_unpad)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114,114,114))
    return img
```

**Normalizacja:**
```python
# YOLOv8 default: 0-1
img = img / 255.0

# Alternatywnie: ImageNet stats
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = (img - mean) / std
```

---

## 3.2 Podział na siatkę (Grid)

### Jak działa siatka

**Zasady:**
1. Obraz dzielony na S×S komórek
2. Komórka jest "odpowiedzialna" za obiekt, jeśli **środek** obiektu w niej leży
3. W nowszych YOLO: jedna komórka może wykryć wiele obiektów

**Przykład dla YOLOv8 (input 640×640):**

| Skala | Grid | Komórek | Stride | Pixels/cell |
|-------|------|---------|--------|-------------|
| P3 | 80×80 | 6,400 | 8 | 8×8 |
| P4 | 40×40 | 1,600 | 16 | 16×16 |
| P5 | 20×20 | 400 | 32 | 32×32 |
| **Razem** | - | **8,400** | - | - |

**Mapowanie środka obiektu na komórkę:**
```python
# Obiekt ze środkiem w (320, 240) pikseli

# Skala P3 (stride 8):
cell_x = 320 // 8 = 40
cell_y = 240 // 8 = 30
# Odpowiedzialna komórka: (40, 30) w siatce 80×80

# Skala P4 (stride 16):
cell_x = 320 // 16 = 20
cell_y = 240 // 16 = 15
# Odpowiedzialna komórka: (20, 15) w siatce 40×40
```

---

## 3.3 Format predykcji

### Co przewiduje każda komórka

**Raw output (przed dekodowaniem):**
```
Dla każdej komórki i każdego anchora/punktu:
[t_x, t_y, t_w, t_h, obj_conf, class_0, class_1, ..., class_C-1]
      ↓
     4      +    1    +            C              = 5 + C wartości
```

**Dla COCO (80 klas):**
- 5 + 80 = 85 wartości na detekcję
- YOLOv8 anchor-free: 4 + 80 = 84 (bez explicit objectness)

### Obliczanie finalnego confidence

**YOLOv3-v5:**
```
class_conf[i] = obj_conf × class_prob[i]
              = P(object) × P(class_i | object)
              = P(class_i ∩ object)
```

**YOLOv8:**
```
# Bez explicit objectness
class_conf[i] = sigmoid(class_logit[i])
```

### Tensor wyjściowy

**Kształt (przed reshape):**
```
Batch × (Anchors × (5 + Classes)) × H × W
```

**Po reshape (dla łatwiejszego przetwarzania):**
```
Batch × (H × W × Anchors) × (5 + Classes)
     = Batch × Num_predictions × Attributes
```

**Przykład YOLOv8:**
```
# 3 skale: P3(80×80), P4(40×40), P5(20×20)
# 80 klas COCO, 1 anchor/punkt (anchor-free)

P3: 1 × 6400 × 84
P4: 1 × 1600 × 84
P5: 1 × 400 × 84
─────────────────
Concat: 1 × 8400 × 84
```

---

## 3.4 Bounding Box Encoding/Decoding

### Dlaczego kodowanie?

**Problem:**
- Sieci neuronowe lepiej przewidują małe wartości (0-1)
- Bezpośrednie współrzędne mogą być niestabilne
- Różne rozmiary obiektów → różne zakresy wartości

**Rozwiązanie:**
- Przewiduj offsety/transformacje względem znanego punktu
- Normalizowane wartości w kontrolowanym zakresie

### Anchor-based decoding (YOLOv3-v5)

```python
def decode_anchor_based(pred, anchors, stride, grid):
    """
    pred: [batch, anchors, 5+classes, H, W]
    anchors: [(p_w1, p_h1), (p_w2, p_h2), ...]
    stride: 8, 16, lub 32
    grid: meshgrid współrzędnych komórek
    """
    # Środek
    b_x = (torch.sigmoid(pred[..., 0]) * 2 - 0.5 + grid[..., 0]) * stride
    b_y = (torch.sigmoid(pred[..., 1]) * 2 - 0.5 + grid[..., 1]) * stride

    # Wymiary
    b_w = (torch.sigmoid(pred[..., 2]) * 2) ** 2 * anchors[..., 0] * stride
    b_h = (torch.sigmoid(pred[..., 3]) * 2) ** 2 * anchors[..., 1] * stride

    # Confidence
    obj_conf = torch.sigmoid(pred[..., 4])

    # Klasy
    class_conf = torch.sigmoid(pred[..., 5:])

    return b_x, b_y, b_w, b_h, obj_conf, class_conf
```

### Anchor-free decoding (YOLOv8)

**Format: Distance to edges (l, t, r, b)**
- l = odległość do lewej krawędzi
- t = odległość do górnej krawędzi
- r = odległość do prawej krawędzi
- b = odległość do dolnej krawędzi

```python
def decode_anchor_free(pred, stride, grid):
    """
    pred: [batch, 4+classes, H, W]
    Format: [l, t, r, b, class_0, class_1, ...]
    """
    # Distribution Focal Loss decoding
    # pred[..., :4] to rozkład prawdopodobieństw dla l,t,r,b

    # Zdekodowane odległości (po DFL)
    lt = pred[..., :2]  # left, top
    rb = pred[..., 2:4]  # right, bottom

    # Współrzędne środka komórki
    c_x = (grid[..., 0] + 0.5) * stride
    c_y = (grid[..., 1] + 0.5) * stride

    # Finalne współrzędne
    x1 = c_x - lt[..., 0] * stride
    y1 = c_y - lt[..., 1] * stride
    x2 = c_x + rb[..., 0] * stride
    y2 = c_y + rb[..., 1] * stride

    # Klasy
    class_conf = torch.sigmoid(pred[..., 4:])

    return x1, y1, x2, y2, class_conf
```

---

## 3.5 IoU (Intersection over Union)

### Definicja

```
IoU = Area(Intersection) / Area(Union)
    = Area(A ∩ B) / Area(A ∪ B)
    = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

### Implementacja

```python
def calculate_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2] format
    """
    # Współrzędne intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Area of intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Areas of individual boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
```

### Interpretacja IoU

| IoU | Interpretacja |
|-----|---------------|
| 1.0 | Idealne dopasowanie |
| 0.75 | Bardzo dobre dopasowanie |
| 0.5 | Typowy próg "poprawnej" detekcji |
| 0.25 | Słabe dopasowanie |
| 0.0 | Brak pokrycia |

### Zastosowania IoU

1. **Metryka ewaluacji (mAP):**
   - AP@0.5: detekcja poprawna jeśli IoU ≥ 0.5
   - AP@0.75: bardziej rygorystyczny próg
   - AP@0.5:0.95: średnia dla progów 0.5, 0.55, ..., 0.95

2. **Non-Maximum Suppression:**
   - Jeśli IoU > threshold → usuń słabszą detekcję

3. **Label assignment (trening):**
   - Przypisanie predykcji do ground truth

---

## 3.6 Non-Maximum Suppression (NMS)

### Problem

Wiele komórek siatki może wykryć ten sam obiekt → duplikaty

```
Przed NMS:
┌────────────┐
│ ┌──────┐   │  conf=0.95
│ │      │   │
│ │ ┌────┴─┐ │  conf=0.87
│ │ │      │ │
│ └─┤      │ │  conf=0.72
│   └──────┘ │
└────────────┘
3 nakładające się detekcje tego samego obiektu
```

### Algorytm NMS

```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    boxes: [N, 4] - współrzędne bbox
    scores: [N] - confidence scores
    iou_threshold: próg duplikatu
    """
    # Sortuj po confidence (malejąco)
    order = scores.argsort(descending=True)
    keep = []

    while len(order) > 0:
        # Weź najlepszą detekcję
        idx = order[0]
        keep.append(idx)

        # Jeśli to ostatnia, zakończ
        if len(order) == 1:
            break

        # Oblicz IoU z pozostałymi
        ious = calculate_iou(boxes[idx], boxes[order[1:]])

        # Zostaw tylko te z IoU < threshold
        mask = ious < iou_threshold
        order = order[1:][mask]

    return keep
```

### Warianty NMS

**Soft-NMS:**
- Zamiast usuwać, obniża score nakładających się detekcji
- `new_score = score * exp(-iou² / sigma)`
- Lepsze dla nakładających się obiektów

**Batched NMS:**
- NMS osobno dla każdej klasy
- Standard w PyTorch: `torchvision.ops.batched_nms()`

**DIoU-NMS (YOLOv4):**
- Używa DIoU zamiast IoU
- Uwzględnia odległość środków

**NMS-free (YOLOv10):**
- Eliminacja NMS przez one-to-one matching
- Dual label assignment podczas treningu
- Szybsza inferencja (~20% mniej latencji)

---

## 3.7 Confidence Threshold vs IoU Threshold

### Confidence Threshold

**Co kontroluje:** Minimalny confidence score do zachowania detekcji

**Trade-off:**
- **Wyższy** (np. 0.7): mniej false positives, może przegapić obiekty
- **Niższy** (np. 0.1): więcej detekcji, więcej fałszywych alarmów

**Typowe wartości:**
- Produkcja/deployment: 0.25 - 0.5
- Analiza/debugging: 0.1
- Wysoka precyzja: 0.6 - 0.8

### IoU Threshold (NMS)

**Co kontroluje:** Kiedy dwie detekcje są "duplikatem"

**Trade-off:**
- **Wyższy** (np. 0.7): więcej nakładających się boxów (gęste obiekty)
- **Niższy** (np. 0.3): agresywniejsze usuwanie (rzadkie obiekty)

**Typowe wartości:**
- Default: 0.45 - 0.5
- Gęste obiekty (tłum): 0.6 - 0.7
- Rzadkie obiekty: 0.3 - 0.4

### Przykład użycia (Ultralytics)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Standardowe ustawienia
results = model.predict(source='image.jpg', conf=0.25, iou=0.45)

# Wysoka precyzja (mniej FP)
results = model.predict(source='image.jpg', conf=0.5, iou=0.5)

# Wysokie recall (znajdź wszystko)
results = model.predict(source='image.jpg', conf=0.1, iou=0.7)
```

---

# Część 4: Funkcja straty (Loss Function)

## 4.1 Przegląd

### Cel funkcji straty

Mierzy jak bardzo predykcje sieci różnią się od ground truth podczas treningu.
Gradient loss jest używany do aktualizacji wag (backpropagation).

### Składowe Loss YOLO

```
L_total = λ_box · L_box + λ_obj · L_obj + λ_cls · L_cls
```

| Składowa | Zadanie | Waga (YOLOv8) |
|----------|---------|---------------|
| L_box | Lokalizacja bbox | λ_box = 7.5 |
| L_obj | Czy jest obiekt | λ_obj = 1.5 |
| L_cls | Klasyfikacja | λ_cls = 0.5 |

### Dlaczego różne wagi?

- Lokalizacja bbox jest najtrudniejsza → najwyższa waga
- Klasyfikacja jest łatwiejsza → niższa waga
- Balans zapobiega dominacji jednego zadania

---

## 4.2 Localization Loss

### YOLOv1: MSE Loss

```python
L_box = Σ[(x - x̂)² + (y - ŷ)²] + Σ[(√w - √ŵ)² + (√h - √ĥ)²]
```

**Dlaczego sqrt dla w, h?**
- Błąd 10px jest poważniejszy dla małego obiektu (50px) niż dużego (500px)
- Sqrt zmniejsza wpływ absolutnej różnicy

**Problemy MSE:**
1. Nie uwzględnia geometrii bbox jako całości
2. Traktuje x, y, w, h niezależnie
3. Nie jest skalowo-niezmienny

### IoU Loss

```python
L_box = 1 - IoU(pred, target)
```

**Zalety:**
- Bezpośrednio optymalizuje metrykę
- Uwzględnia bbox jako całość
- Skalowo-niezmienny (scale-invariant)

**Problem:**
- Gdy IoU = 0 (boxy się nie przecinają), gradient = 0
- Sieć nie wie w którym kierunku iść

### GIoU (Generalized IoU)

**Autorzy:** Rezatofighi et al., 2019

```python
GIoU = IoU - (|C - (A ∪ B)| / |C|)
```

Gdzie C = najmniejszy prostokąt zawierający oba boxy

```
┌──────────────────────┐  ← C (enclosing box)
│                      │
│   ┌─────┐   ┌─────┐  │
│   │  A  │   │  B  │  │  ← A i B się nie przecinają
│   └─────┘   └─────┘  │
│                      │
└──────────────────────┘
```

**Korzyści:**
- Gradient nawet gdy IoU = 0
- Karze za duży C (zachęca do zbliżania boxów)

### DIoU (Distance IoU)

**Autorzy:** Zheng et al., 2020

```python
DIoU = IoU - (ρ²(b, b_gt) / c²)
```

Gdzie:
- ρ(b, b_gt) = odległość euklidesowa między środkami
- c = przekątna C (enclosing box)

**Korzyści:**
- Bezpośrednio minimalizuje odległość środków
- Szybsza konwergencja niż GIoU
- Stabilniejszy trening

### CIoU (Complete IoU)

```python
CIoU = IoU - (ρ²(b, b_gt) / c²) - α·v
```

Gdzie:
```python
v = (4 / π²) · (arctan(w_gt/h_gt) - arctan(w/h))²  # różnica aspect ratio
α = v / (1 - IoU + v)  # waga adaptacyjna
```

**Korzyści:**
- Uwzględnia: overlap (IoU), odległość środków (DIoU), aspect ratio (v)
- Najbardziej kompletna metryka
- **Domyślnie używane w YOLOv5+**

### Porównanie wizualne

```
              IoU     GIoU    DIoU    CIoU
Overlap        ✓       ✓       ✓       ✓
Non-overlap    ✗       ✓       ✓       ✓
Center dist    ✗       ✗       ✓       ✓
Aspect ratio   ✗       ✗       ✗       ✓
```

### Implementacja CIoU

```python
def ciou_loss(pred_box, target_box):
    """
    pred_box, target_box: [x1, y1, x2, y2]
    """
    # IoU
    inter = box_intersection(pred_box, target_box)
    union = box_union(pred_box, target_box)
    iou = inter / union

    # Enclosing box
    c_x1 = min(pred_box[0], target_box[0])
    c_y1 = min(pred_box[1], target_box[1])
    c_x2 = max(pred_box[2], target_box[2])
    c_y2 = max(pred_box[3], target_box[3])
    c_diag_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    # Distance between centers
    pred_cx = (pred_box[0] + pred_box[2]) / 2
    pred_cy = (pred_box[1] + pred_box[3]) / 2
    target_cx = (target_box[0] + target_box[2]) / 2
    target_cy = (target_box[1] + target_box[3]) / 2
    rho_sq = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2

    # Aspect ratio term
    pred_w = pred_box[2] - pred_box[0]
    pred_h = pred_box[3] - pred_box[1]
    target_w = target_box[2] - target_box[0]
    target_h = target_box[3] - target_box[1]

    v = (4 / math.pi**2) * (
        math.atan(target_w / target_h) - math.atan(pred_w / pred_h)
    )**2

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - rho_sq / c_diag_sq - alpha * v
    loss = 1 - ciou

    return loss
```

---

## 4.3 Objectness Loss

### Zadanie

Nauczyć sieć rozróżniać:
- Komórki zawierające obiekt (positive)
- Komórki tła (negative)

### Binary Cross-Entropy

```python
L_obj = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Gdzie:
- y = 1 jeśli komórka zawiera obiekt, 0 jeśli tło
- ŷ = predykcja sieci (po sigmoid)

### Problem: Class Imbalance

**Statystyki typowego obrazu:**
- ~8400 predykcji (dla YOLOv8)
- ~5-10 pozytywnych (obiekty)
- ~8390 negatywnych (tło)
- Ratio: ~99.9% negative!

**Konsekwencje:**
- Sieć uczy się przewidywać "wszystko to tło"
- Gradient zdominowany przez łatwe negatywne przykłady
- Słaba wydajność dla rzeczywistych obiektów

### Rozwiązania

**1. Różne wagi (YOLOv1-v3):**
```python
L_obj = λ_obj · Σ_pos[...] + λ_noobj · Σ_neg[...]
# λ_obj = 1.0, λ_noobj = 0.5
```

**2. Focal Loss (Lin et al., 2017):**
```python
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

Gdzie:
- p_t = p jeśli y=1, (1-p) jeśli y=0
- α_t = α jeśli y=1, (1-α) jeśli y=0
- γ = focusing parameter (typowo 2)

**Jak działa Focal Loss:**
- Łatwe przykłady (p_t ≈ 1): (1-p_t)^γ ≈ 0 → niska waga
- Trudne przykłady (p_t ≈ 0): (1-p_t)^γ ≈ 1 → wysoka waga
- Skupia się na trudnych przykładach

**Implementacja:**
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    pred: [N] - predykcje (po sigmoid)
    target: [N] - labels (0 lub 1)
    """
    # Binary cross entropy
    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # p_t
    p_t = pred * target + (1 - pred) * (1 - target)

    # Focal weight
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    # Final loss
    loss = alpha_t * focal_weight * bce
    return loss.mean()
```

---

## 4.4 Classification Loss

### Multi-class classification

**Softmax (mutually exclusive):**
```python
# Klasy są wzajemnie wykluczające się
# Jeden obiekt = jedna klasa
L_cls = CrossEntropyLoss(pred, target)
```

**Binary CE per class (YOLOv3+):**
```python
# Klasy są niezależne
# Jeden obiekt może mieć wiele klas (np. "person" i "woman")
L_cls = Σ_i[BCELoss(pred_i, target_i)]
```

### Label Smoothing

**Problem:** Over-confidence
- Sieć uczy się dawać 100% na poprawną klasę
- Słaba kalibracja, wrażliwość na szum

**Rozwiązanie:**
```python
# Zamiast y = [0, 0, 1, 0, 0] (one-hot)
# Używaj y_smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
# gdzie ε = 0.1, K = liczba klas

def label_smoothing(target, num_classes, smoothing=0.1):
    # target: [batch_size] - indeksy klas
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / num_classes

    one_hot = torch.zeros_like(pred)
    one_hot.fill_(smoothing_value)
    one_hot.scatter_(1, target.unsqueeze(1), confidence)

    return one_hot
```

### Distribution Focal Loss (DFL)

**YOLOv8 innowacja dla bbox regression:**

Zamiast przewidywać jedną wartość (np. odległość do krawędzi), przewiduj **rozkład prawdopodobieństwa** nad możliwymi wartościami.

```python
# Zamiast: pred = 12.5 (odległość)
# Używaj: pred = [p_0, p_1, ..., p_15] rozkład nad [0, 1, ..., 15]
# Finalna wartość = Σ_i(i * softmax(pred_i))
```

**Korzyści:**
- Lepsza lokalizacja granic obiektów
- Reprezentacja niepewności
- Gładsze gradienty

**Implementacja:**
```python
def dfl_loss(pred_dist, target):
    """
    pred_dist: [batch, 16] - rozkład nad 16 wartościami
    target: [batch] - ground truth wartość (float)
    """
    # Discretize target
    target_left = target.floor().long()
    target_right = target_left + 1
    weight_right = target - target_left.float()
    weight_left = 1 - weight_right

    # Cross entropy dla lewego i prawego sąsiada
    loss = (
        F.cross_entropy(pred_dist, target_left, reduction='none') * weight_left +
        F.cross_entropy(pred_dist, target_right, reduction='none') * weight_right
    )
    return loss.mean()
```

---

## 4.5 Task-Aligned Learning (TAL)

### Problem: Label Assignment

Podczas treningu musimy przypisać:
- Predykcje do ground truth obiektów
- Które komórki są positive (uczą się wykrywać)
- Które są negative (uczą się że to tło)

### Tradycyjne podejścia

**IoU-based (YOLOv3-v5):**
```python
# Predykcja jest positive jeśli:
# IoU(prediction, ground_truth) > threshold
# i anchor najlepiej pasuje do obiektu
```

**Problemy:**
- Threshold jest arbitralny
- Nie uwzględnia jakości klasyfikacji
- Może przypisać dużo słabych predykcji

### Task-Aligned Assigner (YOLOv8)

**Idea:** Przypisanie powinno uwzględniać ZARÓWNO klasyfikację JAK I lokalizację

**Metric alignment:**
```python
alignment_metric = cls_score^α × IoU^β
# α = 1.0, β = 6.0 (domyślnie)
```

**Algorytm:**
1. Dla każdego ground truth, oblicz alignment_metric dla wszystkich predykcji
2. Wybierz top-k predykcji z najwyższym alignment_metric
3. Te predykcje są positive dla tego ground truth

**Implementacja uproszczona:**
```python
def task_aligned_assigner(pred_scores, pred_bboxes, gt_bboxes, gt_labels, topk=13):
    """
    pred_scores: [num_anchors, num_classes]
    pred_bboxes: [num_anchors, 4]
    gt_bboxes: [num_gt, 4]
    gt_labels: [num_gt]
    """
    num_anchors = pred_bboxes.shape[0]
    num_gt = gt_bboxes.shape[0]

    # Classification scores for GT classes
    cls_scores = pred_scores[:, gt_labels]  # [num_anchors, num_gt]

    # IoU between predictions and GTs
    ious = box_iou(pred_bboxes, gt_bboxes)  # [num_anchors, num_gt]

    # Alignment metric
    align_metric = cls_scores.pow(1.0) * ious.pow(6.0)

    # Top-k per GT
    topk_mask = torch.zeros_like(align_metric, dtype=torch.bool)
    for j in range(num_gt):
        _, topk_idxs = align_metric[:, j].topk(topk)
        topk_mask[topk_idxs, j] = True

    # Resolve conflicts (anchor assigned to multiple GTs)
    # Keep assignment with highest alignment metric
    ...

    return assigned_gt_idx, assigned_labels
```

---

## 4.6 Pełny Loss YOLOv8

```python
class YOLOv8Loss:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5

    def __call__(self, pred, targets):
        """
        pred: dict z predykcjami
        targets: ground truth
        """
        # Rozdziel predykcje
        pred_distri = pred['box_distribution']  # dla DFL
        pred_scores = pred['cls_scores']

        # Task-Aligned Assignment
        assigned_targets = self.assigner(pred, targets)

        # Box Loss (CIoU + DFL)
        loss_box = self.ciou_loss(pred_bboxes, assigned_bboxes)
        loss_dfl = self.dfl_loss(pred_distri, assigned_distri)

        # Classification Loss (BCE)
        loss_cls = self.bce(pred_scores, assigned_labels).mean()

        # Total Loss
        total_loss = (
            self.box_weight * loss_box +
            self.dfl_weight * loss_dfl +
            self.cls_weight * loss_cls
        )

        return total_loss
```

---

## 4.7 Podsumowanie ewolucji Loss

| Wersja | Box Loss | Obj Loss | Cls Loss | Assignment |
|--------|----------|----------|----------|------------|
| YOLOv1 | MSE | MSE | MSE | Grid-based |
| YOLOv2 | MSE | BCE | BCE | Anchor IoU |
| YOLOv3 | MSE | BCE | BCE | Anchor IoU |
| YOLOv4 | CIoU | BCE | BCE | Anchor IoU |
| YOLOv5 | CIoU | BCE + Focal | BCE | Anchor IoU |
| YOLOX | IoU/GIoU | Focal | BCE | SimOTA |
| YOLOv8 | CIoU + DFL | - (implicit) | BCE | TAL |

---

## Źródła i literatura (Części 2-4)

1. Lin, T.Y. et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
2. Liu, S. et al. (2018). Path Aggregation Network for Instance Segmentation. CVPR.
3. Wang, C.Y. et al. (2020). CSPNet: A New Backbone that can Enhance Learning Capability of CNN. CVPRW.
4. Rezatofighi, H. et al. (2019). Generalized Intersection over Union. CVPR.
5. Zheng, Z. et al. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. AAAI.
6. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV.
7. Ge, Z. et al. (2021). YOLOX: Exceeding YOLO Series in 2021. arXiv.
8. Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
9. Wang, A. et al. (2024). YOLOv10: Real-Time End-to-End Object Detection. arXiv.
