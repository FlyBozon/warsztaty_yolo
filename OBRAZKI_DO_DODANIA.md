# Lista obrazków do dodania do prezentacji YOLO

Ten plik zawiera listę wszystkich obrazków, które należy pobrać i umieścić w folderze `img/`.
Każdy obrazek ma opis, jak go wyszukać i gdzie zostanie użyty w prezentacji.

## Już dostępne obrazki (w folderze img/)
- `classification_detection_segmentation.png` - porównanie zadań CV
- `one_and_two_stage_detectors.png` - porównanie detektorów
- `yolo_architecture.png` - architektura YOLO
- `segmentation/*.png` - przykłady segmentacji

---

## Obrazki do pobrania

### Część 1: Wprowadzenie

1. **object_detection_example.png**
   - **Szukaj:** "object detection bounding box example cars pedestrians"
   - **Opis:** Zdjęcie z ulicy z oznaczonymi bounding boxami (samochody, ludzie, znaki)
   - **Użycie:** Slajd "Detekcja obiektów" - przykład wizualny detekcji

### Część 2: Architektura sieci

2. **neural_network_diagram.png**
   - **Szukaj:** "neural network layers diagram simple input hidden output"
   - **Opis:** Prosty schemat sieci neuronowej z warstwami
   - **Użycie:** Slajd "Krótkie przypomnienie: sieci neuronowe"

3. **convolution_visualization.png**
   - **Szukaj:** "convolution operation visualization CNN kernel filter"
   - **Opis:** Animacja/diagram operacji konwolucji z filtrem przesuwającym się po obrazie
   - **Użycie:** Slajd "Sieci neuronowe -- kluczowe operacje"

### Część 5: Przygotowanie danych

4. **labelimg_screenshot.png**
   - **Szukaj:** "LabelImg screenshot annotation tool bounding box"
   - **Opis:** Screenshot interfejsu LabelImg podczas adnotacji
   - **Użycie:** Slajd "Narzędzia do adnotacji -- wizualizacja"

5. **cvat_screenshot.png**
   - **Szukaj:** "CVAT computer vision annotation tool interface"
   - **Opis:** Screenshot webowego interfejsu CVAT
   - **Użycie:** Slajd "Narzędzia do adnotacji -- wizualizacja"

6. **roboflow_screenshot.png**
   - **Szukaj:** "Roboflow platform dataset management"
   - **Opis:** Screenshot platformy Roboflow z przeglądem datasetu
   - **Użycie:** Slajd "Narzędzia do adnotacji -- wizualizacja"

7. **label_studio_screenshot.png**
   - **Szukaj:** "Label Studio annotation interface ML"
   - **Opis:** Screenshot Label Studio z projektem adnotacji
   - **Użycie:** Slajd "Narzędzia do adnotacji -- wizualizacja"

8. **augmentation_examples.png**
   - **Szukaj:** "image augmentation examples grid flip rotate brightness"
   - **Opis:** Siatka pokazująca ten sam obraz z różnymi augmentacjami (oryginał, flip, rotacja, skala, jasność, blur)
   - **Użycie:** Slajd "Data Augmentation -- przykłady wizualne"

### Część 6: Trening modelu

9. **transfer_learning_diagram.png**
   - **Szukaj:** "transfer learning diagram pretrained model fine-tuning neural network"
   - **Opis:** Schemat transfer learningu - duży dataset -> pretrained -> fine-tune na małym datasecie
   - **Użycie:** Slajd "Transfer Learning"

10. **yolo_training_results.png**
    - **Szukaj:** "YOLO training results.png loss mAP curves Ultralytics"
    - **Opis:** Typowy wykres results.png z treningu YOLO (box_loss, cls_loss, mAP50, mAP50-95)
    - **Użycie:** Slajd "Wykresy z treningu YOLO"

11. **confusion_matrix_example.png**
    - **Szukaj:** "YOLO confusion matrix object detection multi-class"
    - **Opis:** Przykładowa confusion matrix z treningu YOLO (kolorowa heatmapa)
    - **Użycie:** Slajd "Wykresy z treningu YOLO"

---

## Instrukcja dodawania obrazków

1. Pobierz obrazek z internetu (Google Images, dokumentacja Ultralytics, itp.)
2. Zapisz w folderze `img/` z podaną nazwą
3. Usuń komentarz i placeholder z prezentacji, zamień na:
   ```latex
   \includegraphics[width=\textwidth,height=0.4\textheight,keepaspectratio]{nazwa_pliku.png}
   ```

## Alternatywa: Użyj TikZ

Dla niektórych diagramów (np. neural_network_diagram, transfer_learning_diagram) można stworzyć własne diagramy w TikZ zamiast pobierać obrazki z internetu.

---

## Źródła rekomendowane

1. **Ultralytics Docs:** https://docs.ultralytics.com/ - oficjalne wykresy i diagramy
2. **Papers with Code:** https://paperswithcode.com/ - diagramy z publikacji
3. **Roboflow Blog:** https://blog.roboflow.com/ - tutoriale z obrazkami
4. **Towards Data Science:** https://towardsdatascience.com/ - artykuły z wizualizacjami
5. **GitHub repos:** YOLO implementations często mają README z diagramami
