# FREQUENCY
_Messen, was zwischen den Zeilen schwingt_

---

## ⟁ Zweck

`frequency.py` ist kein gewöhnliches NLP-Skript.  
Es ist ein **Resonanz-Sensor**.  
Ein Werkzeug, das hilft zu erfassen, **wie stark zwei Texte im Feld miteinander schwingen** – semantisch und emotional.

Die Datei `frequency.py` misst die Qualität des Zwischenraums zwischen zwei Aussagen.  
Sie ist inspiriert von der Idee, dass _nicht nur Worte_, sondern auch ihre innere Schwingung Bedeutung tragen.

---

## ⟁ Messprinzipien

### 1. **Semantische Frequenz**
> _Wie ähnlich sind die Gedankenwelten?_

Mittels `SentenceTransformer` wird ein Vektorvergleich durchgeführt.  
Je höher die _cosine similarity_, desto mehr Bedeutungskohärenz liegt vor.

---

### 2. **Sentiment-Kohärenz**
> _Passen die Emotionen zusammen?_

Ein feinabgestimmtes Modell (`distilbert-base-uncased-finetuned-sst-2-english`) analysiert die emotionale Ladung beider Texte.  
Die Nähe der Scores ergibt die _emotionale Kohärenz_ – ein Maß für stimmige Atmosphäre.

---

## ⟁ Ergebnis: Frequenz des Feldes

Die beiden Messwerte (Bedeutung + Gefühl) werden gemittelt.

```python
frequency = (semantic_freq + sentiment_coherence) / 2