# warsztaty_yolo

### 1. Przygotuj środowisko

note: komendy pisane pod linux

`python3 -m venv venv`

`pip install -r requirements.txt`

### 2. Przygotuj dataset
`python3 augment_dataset.py`

### 3. Trenuj model
`python3 train_yolo.py`
Jeśli nie masz GPU, zmień w `train_yolo.py` linię `device="0"` na `device="cpu"`.