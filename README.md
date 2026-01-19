# warsztaty_yolo

### 1. Przygotuj środowisko

note: komendy pisane pod linux

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

### 2. Odpalenie pretrenowanego modelu (opcjonalne)
korzystaj z poleceń na github yolov8 https://github.com/autogyro/yolo-V8

uruchomienie predykcji na obrazie testowym:
`yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"`

uruchomienie real time na obrazie z kamerki na komputerze:
`yolo predict model=yolov8n.pt source=0 show=True`

## Trening własnego modelu:
### 3. Przygotuj dataset
`python3 augment_dataset.py`

### 4. Trenuj model
`python3 train_yolo.py`
Jeśli nie masz GPU, zmień w `train_yolo.py` linię `device="0"` na `device="cpu"`.


