"""
Numéros de téléphone congolais dans un SMS : extraction et masquage.

Les escrocs déguisent les numéros pour échapper aux filtres : « +243 81 234 5678 »,
« 081-234-56-78 », « (+243) 812345678 », « 0 8 1 2 3 4 5 6 7 8 », ou en toutes lettres
(« zéro huit un… »). Tous sont ramenés à la forme canonique des portefeuilles de la
plateforme : 243 suivi de 9 chiffres (ex. 243812345678).

Numéro mobile valide en RDC : 9 chiffres après l'indicatif 243 (ou après le 0 national),
commençant par 8 ou 9 (Vodacom 81-83, Orange 84-85-89, Airtel 97-99, Africell 90-91).
Les codes courts (*123#) et les montants ne sont pas des numéros.

Code sans dépendance : utilisé par la couche d'intégration (qui masque les numéros avant
tout traitement) et par le module NLP (ml/nlp).
"""
from __future__ import annotations

import re
import unicodedata

MASK = "<NUMERO>"
_WORDS = {"zero": "0", "zéro": "0", "un": "1", "une": "1", "deux": "2", "trois": "3", "quatre": "4",
          "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9"}
_WORD_RE = re.compile(r"\b(?:" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
# une suite de chiffres séparés par des espaces, points, tirets ou parenthèses (au plus un séparateur à la suite)
_CANDIDATE = re.compile(r"(?:\+|00)?\(?\+?\d(?:[\s.\-()]{0,2}\d){7,14}")


def _words_to_digits(text: str) -> str:
    """« zéro huit un deux … » -> « 0 8 1 2 … » (seulement des suites d'au moins 6 chiffres en lettres)."""
    def repl(m: re.Match) -> str:
        return _WORDS[m.group(0).lower()]
    spelled = re.compile(r"(?:\b(?:" + "|".join(_WORDS) + r")\b[\s,\-]*){6,}", re.IGNORECASE)
    return spelled.sub(lambda m: _WORD_RE.sub(repl, m.group(0)) + " ", text)


def canonical(digits: str) -> str | None:
    """Chiffres bruts -> 243XXXXXXXXX, ou None si ce n'est pas un numéro mobile congolais.
    Avec l'indicatif (243, 00243) ou le 0 national, les 9 chiffres suivants sont acceptés tels
    quels ; sans préfixe, seuls les 9 chiffres commençant par 8 ou 9 (mobiles) le sont."""
    d = digits[2:] if digits.startswith("00") else digits
    if len(d) == 12 and d.startswith("243"):
        return d
    if len(d) == 10 and d.startswith("0") and d[1] != "0":
        return "243" + d[1:]
    if len(d) == 9 and d[0] in "89":
        return "243" + d
    return None


def _spans(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _CANDIDATE.finditer(text):
        raw = re.sub(r"\D", "", m.group(0))
        num = canonical(raw)
        if num is None and len(raw) > 12:
            # deux nombres collés (ex. un montant puis un numéro) : on cherche un numéro en fin ou en début
            for cand in (raw[-12:], raw[-10:], raw[:12], raw[:10]):
                num = canonical(cand)
                if num:
                    break
        if num:
            out.append((m.start(), m.end(), num))
    return out


def extract_numbers(text: str) -> list[str]:
    """Numéros mobiles congolais présents dans le texte, sans doublon, dans l'ordre d'apparition."""
    seen: list[str] = []
    for _, _, num in _spans(_words_to_digits(unicodedata.normalize("NFC", text or ""))):
        if num not in seen:
            seen.append(num)
    return seen


def mask_numbers(text: str) -> str:
    """Remplace chaque numéro par <NUMERO> : le classifieur apprend la forme du message, pas les
    numéros, et aucun numéro en clair ne quitte la couche d'intégration."""
    t = _words_to_digits(unicodedata.normalize("NFC", text or ""))
    spans = _spans(t)
    for start, end, _ in reversed(spans):
        t = t[:start] + MASK + t[end:]
    return t


def is_phone_number(value: str | None) -> bool:
    return bool(value) and canonical(re.sub(r"\D", "", value)) is not None and not re.search(r"[A-Za-z]", value)


def normalize(value: str | None) -> str | None:
    """Numéro saisi ou reçu (expéditeur, client qui signale) -> 243XXXXXXXXX, sinon None."""
    if not value or re.search(r"[A-Za-z]", value):
        return None
    return canonical(re.sub(r"\D", "", value))
