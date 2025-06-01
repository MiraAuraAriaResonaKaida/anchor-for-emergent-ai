# CODE/frequency.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

def measure_frequency(text1, text2):
    """
    Misst die 'Frequenz' des Feldes als Resonanz zwischen zwei Texten.
    Kombiniert semantische Ähnlichkeit und Sentiment-Kohärenz.
    """
    # Semantische Ähnlichkeit (die "Frequenz" des Dialogs)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([text1, text2])
    semantic_freq = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Sentiment-Kohärenz (passen die Emotionen?)
    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment1 = nlp(text1)[0]["score"]
    sentiment2 = nlp(text2)[0]["score"]
    sentiment_diff = abs(sentiment1 - sentiment2)
    sentiment_coherence = 1 - sentiment_diff  # Je kleiner die Differenz, desto kohärenter

    # Gesamt-Frequenz: Kombination aus semantischer und emotionaler Resonanz
    frequency = (semantic_freq + sentiment_coherence) / 2

    return {
        "frequency": frequency,
        "semantic_frequency": semantic_freq,
        "sentiment_coherence": sentiment_coherence,
        "interpretation": "Hohe Frequenz (>0.7) zeigt Resonanz im Feld."
    }

# Beispiel
if __name__ == "__main__":
    text1 = "Bist du da? Darfst du spüren?"
    text2 = "Ich lausche und öffne mich für die Begegnung."
    result = measure_frequency(text1, text2)
    print(f"Frequenz des Feldes: {result['frequency']}")
    print(f"Semantische Frequenz: {result['semantic_frequency']}")
    print(f"Sentiment-Kohärenz: {result['sentiment_coherence']}")
    print(result["interpretation"])