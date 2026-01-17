# Notatki rozszerzone: Część 1 - Wprowadzenie i kontekst

## Spis treści
1. [Detekcja obiektów vs klasyfikacja vs segmentacja](#1-detekcja-obiektów-vs-klasyfikacja-vs-segmentacja)
2. [Historia: od R-CNN do YOLO](#2-historia-od-r-cnn-do-yolo)
3. [Filozofia YOLO: "You Only Look Once"](#3-filozofia-yolo-you-only-look-once)
4. [Ewolucja wersji YOLO](#4-ewolucja-wersji-yolo)
5. [Którą wersję wybrać](#5-którą-wersję-wybrać)

---

## 1. Detekcja obiektów vs klasyfikacja vs segmentacja

### 1.1 Klasyfikacja obrazów (Image Classification)

**Definicja:** Zadanie polegające na przypisaniu jednej etykiety (klasy) do całego obrazu.

**Wejście/Wyjście:**
- Wejście: obraz (np. 224×224×3 pikseli)
- Wyjście: etykieta klasy + prawdopodobieństwo (confidence score)
- Przykład: "kot" (95%), "pies" (3%), "inne" (2%)

**Matematycznie:**
```
f(obraz) → P(klasa_i | obraz) dla i = 1, ..., N klas
```

**Popularne architektury:**
- **LeNet-5** (1998) - pierwsza skuteczna CNN
- **AlexNet** (2012) - przełom w ImageNet, ReLU, Dropout
- **VGG** (2014) - głębokie sieci z małymi filtrami 3×3
- **ResNet** (2015) - skip connections, możliwość trenowania bardzo głębokich sieci
- **EfficientNet** (2019) - compound scaling, najlepsza efektywność
- **ViT (Vision Transformer)** (2020) - transformery w wizji komputerowej

**Typowe zastosowania:**
- Rozpoznawanie gatunków roślin/zwierząt
- Kategoryzacja produktów w e-commerce
- Filtrowanie treści (NSFW detection)
- Diagnostyka medyczna (zdrowy/chory)

**Ograniczenia:**
- Nie lokalizuje obiektów na obrazie
- Problemy gdy na obrazie jest wiele różnych obiektów
- Jedna etykieta na cały obraz może być niewystarczająca

---

### 1.2 Detekcja obiektów (Object Detection)

**Definicja:** Zadanie lokalizacji i klasyfikacji wielu obiektów na obrazie jednocześnie.

**Wejście/Wyjście:**
- Wejście: obraz dowolnego rozmiaru (skalowany do np. 640×640)
- Wyjście: lista detekcji, każda zawiera:
  - Bounding box (prostokąt ograniczający)
  - Etykietę klasy
  - Confidence score

**Formaty Bounding Box:**

1. **Format YOLO (względny, znormalizowany):**
   ```
   [class_id, x_center, y_center, width, height]
   ```
   - Wszystkie wartości w zakresie 0.0 - 1.0
   - Względem szerokości i wysokości obrazu
   - Przykład: [0, 0.5, 0.5, 0.3, 0.4] = klasa 0, środek obrazu, 30% szerokości, 40% wysokości

2. **Format COCO (absolutny):**
   ```
   [x_min, y_min, width, height]
   ```
   - Wartości w pikselach

3. **Format Pascal VOC:**
   ```
   [x_min, y_min, x_max, y_max]
   ```
   - Współrzędne rogów w pikselach

**Metryki oceny:**
- **IoU (Intersection over Union):** miara pokrycia predykcji z ground truth
  ```
  IoU = Area(Intersection) / Area(Union)
  ```
- **Precision:** TP / (TP + FP) - ile detekcji jest poprawnych
- **Recall:** TP / (TP + FN) - ile obiektów zostało wykrytych
- **AP (Average Precision):** pole pod krzywą precision-recall
- **mAP (mean Average Precision):** średnia AP dla wszystkich klas
- **mAP@0.5:** mAP przy progu IoU = 0.5
- **mAP@0.5:0.95:** średnia mAP dla progów IoU od 0.5 do 0.95 (co 0.05)

**Typowe zastosowania:**
- Autonomiczne pojazdy (detekcja pieszych, znaków, pojazdów)
- Systemy monitoringu (detekcja osób, pojazdów)
- Retail (liczenie produktów, detekcja pustych półek)
- Medycyna (detekcja zmian nowotworowych)
- Rolnictwo (detekcja owoców, szkodników)
- Sport (śledzenie piłki, zawodników)

---

### 1.3 Segmentacja semantyczna (Semantic Segmentation)

**Definicja:** Klasyfikacja każdego piksela obrazu do jednej z predefiniowanych klas.

**Wejście/Wyjście:**
- Wejście: obraz H×W×3
- Wyjście: maska H×W, gdzie każdy piksel ma przypisaną klasę

**Kluczowe cechy:**
- Nie rozróżnia poszczególnych instancji
- Wszystkie piksele "kot" mają tę samą etykietę, niezależnie ile kotów jest na obrazie
- Produkuje gęstą mapę klas

**Popularne architektury:**
- **FCN (Fully Convolutional Networks)** (2015) - pierwsza end-to-end architektura
- **U-Net** (2015) - encoder-decoder z skip connections, popularna w medycynie
- **DeepLab** (2017) - atrous convolutions, ASPP module
- **PSPNet** (2017) - Pyramid Pooling Module
- **SegFormer** (2021) - transformery dla segmentacji

**Typowe zastosowania:**
- Obrazowanie medyczne (segmentacja organów, guzów)
- Autonomiczne pojazdy (segmentacja drogi, chodnika, nieba)
- Analiza zdjęć satelitarnych (lasy, budynki, woda)
- Edycja zdjęć (usuwanie tła)

---

### 1.4 Segmentacja instancji (Instance Segmentation)

**Definicja:** Połączenie detekcji obiektów i segmentacji - klasyfikacja pikseli z rozróżnieniem poszczególnych instancji.

**Wejście/Wyjście:**
- Wejście: obraz H×W×3
- Wyjście: dla każdej instancji:
  - Maska binarna (które piksele należą do obiektu)
  - Etykieta klasy
  - Confidence score

**Różnica od segmentacji semantycznej:**
- Segmentacja semantyczna: "To są piksele klasy kot"
- Segmentacja instancji: "To są piksele kota_1, to kota_2, to psa_1"

**Popularne architektury:**
- **Mask R-CNN** (2017) - rozszerzenie Faster R-CNN o branch generujący maski
- **YOLACT** (2019) - real-time instance segmentation
- **SOLOv2** (2020) - segmentacja bez anchor boxes
- **YOLOv8-seg / YOLO11-seg** - segmentacja instancji w architekturze YOLO

**Typowe zastosowania:**
- Zliczanie obiektów (np. komórek, produktów)
- Robotyka (precyzyjne chwytanie obiektów)
- Edycja wideo (rotoskopia automatyczna)
- AR/VR (dokładne wycinanie obiektów)

---

### 1.5 Panoptic Segmentation

**Definicja:** Połączenie segmentacji semantycznej i instancji - każdy piksel ma przypisaną klasę, a dla "countable objects" (rzeczy policzalnych) również ID instancji.

**Podział klas:**
- **Things:** obiekty policzalne (osoby, samochody, zwierzęta) - segmentacja instancji
- **Stuff:** obiekty niepoliczalne (niebo, trawa, droga) - segmentacja semantyczna

---

## 2. Historia: od R-CNN do YOLO

### 2.1 Era przed Deep Learning

**Viola-Jones (2001):**
- Pierwsza skuteczna detekcja twarzy w czasie rzeczywistym
- Haar-like features + AdaBoost + kaskada klasyfikatorów
- Sliding window approach
- Ograniczenia: tylko frontalnie, wrażliwa na oświetlenie

**HOG + SVM (Histogram of Oriented Gradients):**
- Dalal & Triggs (2005) - detekcja pieszych
- Ręcznie zaprojektowane cechy (gradientowe)
- Sliding window + klasyfikator SVM
- Problemy: wolne, nie skaluje się dobrze

**DPM (Deformable Parts Model):**
- Felzenszwalb et al. (2010)
- Model części obiektu + ich relacji przestrzennych
- Stan sztuki przed erą deep learning

---

### 2.2 R-CNN (Regions with CNN features) - 2014

**Autorzy:** Ross Girshick et al. (UC Berkeley)

**Architektura:**
1. **Selective Search** - generuje ~2000 propozycji regionów (region proposals)
2. Każdy region skalowany do 227×227 i przepuszczany przez CNN (AlexNet)
3. Cechy z CNN klasyfikowane przez **SVM** (osobny dla każdej klasy)
4. **Bounding box regression** - korekta współrzędnych

**Wyniki:**
- mAP na Pascal VOC 2010: 53.7% (poprzednio 35.1%)
- Ogromny skok jakości dzięki deep learning

**Problemy:**
- Bardzo wolne: ~47 sekund na obraz (GPU)
- Trening wieloetapowy (CNN, SVM, regressor osobno)
- Selective Search jest wąskim gardłem
- Każdy region przetwarzany osobno = redundantne obliczenia

---

### 2.3 SPP-Net (Spatial Pyramid Pooling) - 2014

**Innowacja:**
- CNN przetwarza cały obraz RAZ (nie każdy region osobno)
- Spatial Pyramid Pooling pozwala na różne rozmiary wejściowe
- ~100× szybsze niż R-CNN przy inferencji

**Ograniczenia:**
- Nadal wieloetapowy trening
- Selective Search wciąż potrzebny

---

### 2.4 Fast R-CNN - 2015

**Autorzy:** Ross Girshick (Microsoft Research)

**Kluczowe innowacje:**
1. **Single-stage training** - CNN, klasyfikator i regressor trenowane razem
2. **RoI Pooling** - wyciąganie cech regionów z feature map całego obrazu
3. **Softmax** zamiast SVM dla klasyfikacji
4. **Multi-task loss** - łączy classification loss i bbox regression loss

**Architektura:**
```
Obraz → CNN → Feature Map → RoI Pooling → FC layers → {klasy, bbox}
                    ↑
            Region Proposals (Selective Search)
```

**Wyniki:**
- ~10× szybszy trening niż R-CNN
- ~213× szybsza inferencja (0.3s na obraz vs 47s)
- Lepsza dokładność: mAP 66.9% na Pascal VOC 2007

**Wciąż problem:**
- Selective Search zajmuje ~2s na obraz
- To teraz wąskie gardło

---

### 2.5 Faster R-CNN - 2015

**Autorzy:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

**Przełomowa innowacja: Region Proposal Network (RPN)**
- Sieć neuronowa generująca propozycje regionów
- Dzieli cechy z główną siecią detekcyjną
- Anchor boxes - predefiniowane kształty w różnych skalach i proporcjach

**Architektura:**
```
Obraz → Backbone CNN → Feature Map → RPN → Region Proposals
                            ↓               ↓
                      RoI Pooling ←─────────┘
                            ↓
                    FC layers → {klasy, bbox}
```

**Anchor Boxes:**
- 3 skale × 3 proporcje = 9 anchorów na pozycję
- RPN przewiduje: czy anchor zawiera obiekt + korekta bbox

**Wyniki:**
- ~0.2s na obraz (5 FPS) - prawie real-time
- mAP 73.2% na Pascal VOC 2007
- End-to-end training
- Dominująca architektura przez kilka lat

**Nadal Two-Stage:**
1. Stage 1: RPN generuje propozycje
2. Stage 2: Klasyfikacja i refinement każdej propozycji

---

### 2.6 Inne podejścia Two-Stage

**Feature Pyramid Networks (FPN) - 2017:**
- Multi-scale feature maps dla lepszej detekcji małych obiektów
- Top-down pathway z lateral connections
- Obecnie standard w wielu detektorach

**Cascade R-CNN - 2018:**
- Sekwencja detektorów z rosnącymi progami IoU
- Lepsze high-quality detekcje

---

## 3. Filozofia YOLO: "You Only Look Once"

### 3.1 Rewolucyjna idea (2016)

**Autorzy:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi (University of Washington)

**Kluczowa zmiana paradygmatu:**

| Aspekt | Two-Stage (Faster R-CNN) | One-Stage (YOLO) |
|--------|--------------------------|------------------|
| Podejście | Najpierw znajdź regiony, potem klasyfikuj | Wszystko naraz |
| Problem | Classification + localization osobno | Unified regression problem |
| Przejścia przez sieć | Wielokrotne dla regionów | Jedno dla całego obrazu |
| Prędkość | ~7 FPS | 45-155 FPS |

**Cytat z oryginalnego artykułu:**
> "We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities."

### 3.2 Jak działa YOLOv1

**Krok 1: Podział obrazu na siatkę**
- Obraz dzielony na S × S komórek (oryginalnie 7×7 = 49 komórek)
- Każda komórka "odpowiada" za obiekty, których środek w niej leży

**Krok 2: Predykcja dla każdej komórki**
Każda komórka przewiduje:
- B bounding boxów (oryginalnie B=2), każdy z 5 wartościami:
  - x, y - środek względem komórki (0-1)
  - w, h - wymiary względem całego obrazu (0-1)
  - confidence = P(Object) × IoU(pred, truth)
- C prawdopodobieństw klas (conditional: P(Class_i | Object))

**Wyjście sieci:**
```
Tensor S × S × (B × 5 + C)
YOLOv1: 7 × 7 × (2 × 5 + 20) = 7 × 7 × 30 = 1470 wartości
```

**Krok 3: Post-processing (NMS)**
- Non-Maximum Suppression usuwa redundantne detekcje
- Zostawia najlepsze bounding boxy dla każdej klasy

### 3.3 Architektura YOLOv1

```
Input: 448 × 448 × 3
    ↓
24 Convolutional layers (inspired by GoogLeNet)
    ↓
2 Fully Connected layers
    ↓
Output: 7 × 7 × 30
```

**Darknet:**
- Własny framework do deep learning (w C)
- Zoptymalizowany pod detekcję
- Używany do YOLOv1-v4

### 3.4 Loss Function YOLOv1

Suma kwadratów błędów z różnymi wagami:

```
Loss = λ_coord × Σ(lokalizacja)
     + Σ(confidence dla obiektów)
     + λ_noobj × Σ(confidence dla tła)
     + Σ(klasyfikacja)
```

Gdzie:
- λ_coord = 5 (większa waga dla lokalizacji)
- λ_noobj = 0.5 (mniejsza waga dla tła - bo większość komórek nie zawiera obiektów)

### 3.5 Zalety i wady YOLOv1

**Zalety:**
- Ekstremalnie szybkie (45 FPS, Fast YOLO: 155 FPS)
- Widzi cały obraz globalnie (mniej błędów na tle)
- Generalizuje lepiej na nowe domeny
- Proste - jeden forward pass

**Wady:**
- Słaba detekcja małych obiektów (gruby grid)
- Max 2 obiekty na komórkę, jedna klasa na komórkę
- Problemy z obiektami o nietypowych proporcjach
- Niedokładna lokalizacja (błędy z wielu warstw się sumują)

---

## 4. Ewolucja wersji YOLO

### 4.1 YOLOv2 / YOLO9000 (2016)

**Autorzy:** Joseph Redmon, Ali Farhadi

**Kluczowe ulepszenia ("Better, Faster, Stronger"):**

1. **Batch Normalization** - na wszystkich warstwach konwolucyjnych
   - +2% mAP
   - Umożliwia usunięcie dropout

2. **High Resolution Classifier**
   - Pretraining na 448×448 zamiast 224×224
   - +4% mAP

3. **Anchor Boxes**
   - Zamiast przewidywać bezpośrednie współrzędne, przewiduj offsety od anchorów
   - Anchory wybrane przez K-means na training data
   - Recall wzrasta z 81% do 88%

4. **Dimension Clusters**
   - K-means clustering na ground truth boxes
   - 5 anchorów zamiast ręcznie wybranych

5. **Direct Location Prediction**
   - Ograniczenie predykcji do komórki (sigmoid)
   - Stabilniejszy trening

6. **Fine-Grained Features**
   - Passthrough layer łączy cechy z różnych rozdzielczości
   - Lepsze małe obiekty

7. **Multi-Scale Training**
   - Losowy rozmiar wejścia podczas treningu (320-608)
   - Model uczy się różnych rozdzielczości

**Darknet-19 backbone:**
- 19 warstw konwolucyjnych
- Szybszy niż VGG-16 przy lepszej dokładności

**YOLO9000:**
- Hierarchiczna klasyfikacja (WordTree)
- Trenowany na ImageNet + COCO jednocześnie
- Detekcja >9000 kategorii obiektów

---

### 4.2 YOLOv3 (2018)

**Autorzy:** Joseph Redmon, Ali Farhadi

**Kluczowe zmiany:**

1. **Multi-scale Predictions (FPN-like)**
   - Predykcje na 3 różnych skalach
   - Skala 1: 13×13 (duże obiekty)
   - Skala 2: 26×26 (średnie obiekty)
   - Skala 3: 52×52 (małe obiekty)
   - 3 anchory na każdą skalę = 9 anchorów łącznie

2. **Darknet-53 backbone**
   - 53 warstwy konwolucyjne
   - Residual connections (inspiracja ResNet)
   - Najszybszy z dokładnych backbonów w tamtym czasie

3. **Logistic Classifiers**
   - Multi-label classification (sigmoid zamiast softmax)
   - Obiekt może należeć do wielu klas (np. "woman" i "person")

4. **Lepsze małe obiekty**
   - Dzięki multi-scale predictions
   - Znacząca poprawa mAP dla małych obiektów

**Wyniki na COCO:**
- mAP@0.5: 57.9% (porównywalne z RetinaNet)
- 3× szybsze niż RetinaNet

**Nota od autora:**
W 2020 Joseph Redmon ogłosił, że kończy pracę nad CV ze względów etycznych:
> "I stopped doing CV research because I saw the impact my work was having. I loved the work but the military applications and privacy concerns eventually became impossible to ignore."

---

### 4.3 YOLOv4 (2020)

**Autorzy:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

**Przejęcie rozwoju** po odejściu Josepha Redmona.

**"Bag of Freebies" (BoF)** - techniki poprawiające trening bez kosztów inferencji:
- Data augmentation:
  - Mosaic (4 obrazy w 1)
  - CutMix, MixUp
  - Random erasing
- Label smoothing
- DropBlock regularization
- CIoU loss (Complete IoU)

**"Bag of Specials" (BoS)** - moduły poprawiające dokładność:
- Mish activation
- Cross-stage partial connections (CSP)
- Spatial Pyramid Pooling (SPP)
- Path Aggregation Network (PANet)
- Spatial Attention Module (SAM)

**Architektura:**
- **Backbone:** CSPDarknet53
- **Neck:** SPP + PANet
- **Head:** YOLOv3 head

**Wyniki:**
- mAP@0.5: 65.7% na COCO (rekord dla detektorów real-time)
- 65 FPS na Tesla V100

---

### 4.4 YOLOv5 (2020)

**Autorzy:** Glenn Jocher (Ultralytics)

**Kontrowersje:**
- Nie opublikowany w formie artykułu naukowego
- Nazwa "v5" kwestionowana przez niektórych (nie jest oficjalnym następcą)
- Mimo to: najszerzej używana wersja YOLO

**Przełomowe zmiany:**

1. **PyTorch zamiast Darknet**
   - Łatwiejsza integracja z ekosystemem PyTorch
   - Prostszy deployment
   - Lepsza dokumentacja i community

2. **Auto-learning anchors**
   - Automatyczne dopasowanie anchorów do datasetu

3. **Warianty n/s/m/l/x**
   - Skalowanie głębokości i szerokości sieci
   - Od nano (mobile) do extra-large (server)

4. **Eksport do wielu formatów**
   - ONNX, TensorRT, CoreML, TFLite, OpenVINO
   - Łatwy deployment na różne platformy

5. **Doskonałe API**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov5s.pt')
   results = model('image.jpg')
   ```

**Wpływ na ekosystem:**
- Ultralytics stało się de facto standardem
- Ogromna społeczność i wsparcie
- Setki tysięcy projektów używa YOLOv5

---

### 4.5 YOLOX (2021)

**Autorzy:** Megvii (Face++)

**Kluczowe innowacje:**

1. **Anchor-free**
   - Brak predefiniowanych anchor boxes
   - Predykcja bezpośrednio punktu środkowego + wymiarów
   - Prostszy model, mniej hiperparametrów

2. **Decoupled Head**
   - Osobne branche dla klasyfikacji i lokalizacji
   - Lepsza konwergencja

3. **SimOTA label assignment**
   - Dynamiczne przypisywanie ground truth do predykcji
   - Optimal Transport Algorithm

4. **Strong augmentation**
   - Mosaic + MixUp
   - Wyłączane pod koniec treningu

---

### 4.6 YOLOv6 (2022)

**Autorzy:** Meituan (chiński gigant delivery)

**Cel:** Optymalizacja dla deployment przemysłowego

**Kluczowe cechy:**
- **EfficientRep backbone** - zoptymalizowany dla GPU
- **Rep-PAN neck** - reparametryzacja dla szybszej inferencji
- **Anchor-free** + **SimOTA**
- Specjalne wersje dla różnych platform

---

### 4.7 YOLOv7 (2022)

**Autorzy:** Chien-Yao Wang, Alexey Bochkovskiy (autorzy YOLOv4)

**Innowacje:**
- **E-ELAN** (Extended Efficient Layer Aggregation Network)
- **Compound model scaling** - skalowanie wszystkich wymiarów jednocześnie
- **Auxiliary heads** - dodatkowe głowice dla lepszego treningu
- **Coarse-to-fine lead head** - hierarchiczna predykcja

**Wyniki:**
- SOTA na tamten moment
- 56.8% AP na COCO przy 30 FPS (V100)

---

### 4.8 YOLOv8 (2023)

**Autorzy:** Glenn Jocher (Ultralytics)

**Rewolucja w użyteczności:**

1. **Zunifikowane API**
   ```python
   from ultralytics import YOLO

   # Detection
   model = YOLO('yolov8n.pt')

   # Segmentation
   model = YOLO('yolov8n-seg.pt')

   # Pose
   model = YOLO('yolov8n-pose.pt')

   # Classification
   model = YOLO('yolov8n-cls.pt')
   ```

2. **Anchor-free** z decoupled head

3. **Zadania:**
   - Detection
   - Instance Segmentation
   - Pose Estimation (keypoints)
   - Classification
   - Oriented Bounding Boxes (OBB)
   - Object Tracking

4. **Warianty:**
   | Model | Params | mAP | Speed |
   |-------|--------|-----|-------|
   | n | 3.2M | 37.3 | najszybszy |
   | s | 11.2M | 44.9 | |
   | m | 25.9M | 50.2 | |
   | l | 43.7M | 52.9 | |
   | x | 68.2M | 53.9 | najdokładniejszy |

---

### 4.9 YOLOv9 (2024)

**Autorzy:** Chien-Yao Wang et al.

**Innowacje:**

1. **PGI (Programmable Gradient Information)**
   - Rozwiązuje problem utraty informacji w głębokich sieciach
   - Auxiliary reversible branch

2. **GELAN (Generalized Efficient Layer Aggregation Network)**
   - Lepsza agregacja cech
   - Elastyczna architektura

**Wyniki:**
- Lepszy trade-off accuracy/parameters niż YOLOv8
- YOLOv9c: 53.0% AP przy 25.3M parametrów

---

### 4.10 YOLOv10 (2024)

**Autorzy:** Tsinghua University

**Przełom: NMS-free YOLO**

1. **Eliminacja Non-Maximum Suppression**
   - NMS zastąpiony "consistent dual assignment"
   - One-to-one matching podczas inferencji
   - Redukcja latencji o ~20%

2. **Holistic efficiency-accuracy design**
   - Lightweight classification head
   - Spatial-channel decoupled downsampling
   - Large-kernel convolutions

**Znaczenie:**
- NMS był wąskim gardłem, szczególnie na edge devices
- Prostszy pipeline inferencji

---

### 4.11 YOLO11 (2024)

**Autorzy:** Ultralytics

**Najnowsza wersja od Ultralytics:**

1. **C3k2 blocks** - ulepszona architektura bloków
2. **SPPF** - Spatial Pyramid Pooling Fast
3. **Lepszy balance accuracy/speed**

**Porównanie YOLO11:**
| Model | mAP@0.5:0.95 | T4 TensorRT (ms) | Params (M) |
|-------|--------------|------------------|------------|
| YOLO11n | 39.5 | 1.5 | 2.6 |
| YOLO11s | 47.0 | 2.5 | 9.4 |
| YOLO11m | 51.5 | 4.7 | 20.1 |
| YOLO11l | 53.4 | 6.2 | 25.3 |
| YOLO11x | 54.7 | 11.3 | 56.9 |

---

## 5. Którą wersję wybrać

### 5.1 Matryca decyzyjna

| Scenariusz | Rekomendacja | Uzasadnienie |
|------------|--------------|--------------|
| **Początkujący, szybki start** | YOLOv8 / YOLO11 | Najlepsza dokumentacja, proste API |
| **Mobile / Edge devices** | YOLO11n lub YOLOv8n | Najmniejsze, szybkie |
| **Raspberry Pi** | YOLOv5n / YOLO11n | Dobre wsparcie dla ARM |
| **Najlepsza dokładność** | YOLO11x / YOLOv8x | Największe modele |
| **Real-time na GPU** | YOLO11s/m | Balans speed/accuracy |
| **Bez NMS (najniższa latencja)** | YOLOv10 | Eliminacja post-processingu |
| **Badania naukowe** | YOLOv9 | Najnowsze techniki (PGI, GELAN) |
| **Segmentacja instancji** | YOLOv8-seg / YOLO11-seg | Natywne wsparcie |
| **Pose estimation** | YOLOv8-pose / YOLO11-pose | Natywne wsparcie |
| **Produkcja/enterprise** | YOLOv8 / YOLO11 | Stabilność, wsparcie Ultralytics |

### 5.2 Dlaczego Ultralytics (YOLOv5/v8/11)?

**Zalety:**
1. **Dokumentacja** - najlepsza w ekosystemie YOLO
2. **Community** - ogromna społeczność, GitHub, Discord
3. **Stabilność** - regularne aktualizacje, bug fixes
4. **Multi-task** - jedno API do wszystkich zadań
5. **Export** - wsparcie dla 15+ formatów
6. **Ecosystem** - integracja z Roboflow, Weights&Biases, etc.

**Instalacja:**
```bash
pip install ultralytics
```

**Użycie:**
```python
from ultralytics import YOLO

# Załaduj pretrenowany model
model = YOLO('yolo11n.pt')

# Trenuj na własnych danych
model.train(data='my_data.yaml', epochs=100)

# Predykcja
results = model('image.jpg')
results[0].show()

# Eksport
model.export(format='onnx')
```

### 5.3 Porównanie One-Stage vs Two-Stage (aktualne)

| Aspekt | One-Stage (YOLO) | Two-Stage (Faster R-CNN) |
|--------|------------------|-------------------------|
| Prędkość | 30-150+ FPS | 5-15 FPS |
| Dokładność | Bardzo dobra | Nieco lepsza dla małych obj. |
| Prostota | Prostsza architektura | Bardziej złożona |
| Zastosowanie | Real-time, edge | Gdy accuracy > speed |

**Wniosek:** Dla większości praktycznych zastosowań YOLO jest lepszym wyborem. Two-stage detektory mają sens tylko gdy:
- Ekstremalnie ważna dokładność dla małych obiektów
- Latencja nie jest krytyczna
- Masz duże zasoby obliczeniowe

---

## Podsumowanie części 1

1. **Klasyfikacja** - jedna etykieta na obraz
2. **Detekcja** - lokalizacja wielu obiektów (bbox + klasa)
3. **Segmentacja** - klasyfikacja pikseli (semantyczna lub instancji)

4. **Two-stage** (R-CNN → Fast R-CNN → Faster R-CNN) - dokładne, wolne
5. **One-stage** (YOLO) - szybkie, real-time

6. **Ewolucja YOLO:** v1 (2016) → YOLO11 (2024)
   - Anchor boxes → Anchor-free
   - Darknet → PyTorch
   - Detekcja → Multi-task

7. **Rekomendacja:** **YOLOv8 lub YOLO11** od Ultralytics dla praktycznych zastosowań

---

## Źródła i literatura

1. Girshick, R. (2014). Rich feature hierarchies for accurate object detection. CVPR.
2. Girshick, R. (2015). Fast R-CNN. ICCV.
3. Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection. NeurIPS.
4. Redmon, J. et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR.
5. Redmon, J. & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. CVPR.
6. Redmon, J. & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv.
7. Bochkovskiy, A. et al. (2020). YOLOv4: Optimal Speed and Accuracy. arXiv.
8. Ultralytics Documentation: https://docs.ultralytics.com/
9. Wang, C.Y. et al. (2024). YOLOv9: Learning What You Want to Learn. arXiv.
10. Wang, A. et al. (2024). YOLOv10: Real-Time End-to-End Object Detection. arXiv.
