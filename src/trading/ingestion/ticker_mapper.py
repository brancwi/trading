"""Extraction de tickers depuis le texte des news générales.

Fait le lien entre une news sans ticker explicite et les entreprises
mentionnées dans le texte via une table de mapping mots-clés → tickers.
"""

import re

# Mapping exhaustif : mots-clés (minuscules) → ticker
KEYWORD_TO_TICKER: dict[str, str] = {
    # Apple
    "apple": "AAPL",
    "iphone": "AAPL",
    "ipad": "AAPL",
    "macbook": "AAPL",
    "tim cook": "AAPL",
    "app store": "AAPL",
    # Tesla
    "tesla": "TSLA",
    "elon musk": "TSLA",
    "model 3": "TSLA",
    "model y": "TSLA",
    "cybertruck": "TSLA",
    # NVIDIA
    "nvidia": "NVDA",
    "geforce": "NVDA",
    "rtx": "NVDA",
    "jensen huang": "NVDA",
    "gpu": "NVDA",
    # Microsoft
    "microsoft": "MSFT",
    "azure": "MSFT",
    "windows": "MSFT",
    "office 365": "MSFT",
    "satya nadella": "MSFT",
    "linkedin": "MSFT",
    # Google / Alphabet
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "sundar pichai": "GOOGL",
    "youtube": "GOOGL",
    "android": "GOOGL",
    "search engine": "GOOGL",
    # Amazon
    "amazon": "AMZN",
    "aws": "AMZN",
    "prime": "AMZN",
    "jeff bezos": "AMZN",
    "andy jassy": "AMZN",
    "e-commerce": "AMZN",
    # Meta
    "meta": "META",
    "facebook": "META",
    "instagram": "META",
    "whatsapp": "META",
    "threads": "META",
    "mark zuckerberg": "META",
    "virtual reality": "META",
    # Palantir
    "palantir": "PLTR",
    "alex karp": "PLTR",
    "big data": "PLTR",
    # Johnson & Johnson
    "johnson & johnson": "JNJ",
    "jnj": "JNJ",
    "pharma": "JNJ",
    # Lockheed Martin
    "lockheed martin": "LMT",
    "f-35": "LMT",
    "defense": "LMT",
    "aeronautics": "LMT",
}


def extract_tickers_from_text(text: str) -> list[str]:
    """Extrait les tickers mentionnés dans un texte via keyword matching.

    Retourne une liste de tickers uniques, triés par pertinence
    (nombre d'occurrences décroissant).
    """
    if not text:
        return []

    text_lower = text.lower()
    ticker_scores: dict[str, int] = {}

    for keyword, ticker in KEYWORD_TO_TICKER.items():
        # Recherche du mot-clé comme mot entier (évite "app" dans "apple")
        # Mais certains mots-clés sont composés ("tim cook")
        pattern = r"\b" + re.escape(keyword) + r"\b"
        matches = len(re.findall(pattern, text_lower))
        if matches:
            ticker_scores[ticker] = ticker_scores.get(ticker, 0) + matches

    # Tri par score décroissant, puis par ticker
    sorted_tickers = sorted(
        ticker_scores.keys(),
        key=lambda t: (-ticker_scores[t], t),
    )
    return sorted_tickers


def map_general_news_to_tickers(title: str, description: str) -> list[str]:
    """Map une news générale (sans ticker) aux tickers potentiellement concernés.

    Combine titre + description pour l'extraction.
    """
    full_text = f"{title} {description or ''}"
    return extract_tickers_from_text(full_text)
