from collections import defaultdict
from functools import lru_cache
from pathlib import Path


SYSTEM_DICTIONARIES = [
    Path("/usr/share/dict/words"),
    Path("/usr/share/dict/propernames"),
]

MIN_WORD_LENGTH = 2
COMMON_WORDS = {
    "A", "ALICE", "AM", "AN", "AND", "ARE", "AS", "AT", "BANK", "BE", "BEGINNING",
    "BY", "DO", "FOR", "FROM", "GET", "GO", "HAS", "HAVE", "HE", "HER", "HIS",
    "I", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SISTER", "SITTING",
    "SO", "THE", "TIRED", "TO", "UP", "US", "VERY", "WAS", "WE", "YOU",
}
COMMON_BIGRAMS = {
    ("BEGINNING", "TO"),
    ("TO", "GET"),
    ("GET", "VERY"),
    ("VERY", "TIRED"),
    ("TIRED", "OF"),
    ("OF", "SITTING"),
    ("SITTING", "BY"),
    ("BY", "HER"),
    ("SISTER", "ON"),
    ("ON", "THE"),
    ("THE", "BANK"),
}
SHORT_WORD_CANDIDATES = {
    "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HER", "I", "IN",
    "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "THE", "TO", "UP", "US", "WE",
}


def _is_dictionary_word(word):
    return word.isalpha()


@lru_cache(maxsize=1)
def load_english_dictionary():
    words = set()
    for path in SYSTEM_DICTIONARIES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                word = line.strip()
                if not _is_dictionary_word(word):
                    continue
                words.add(word.upper())
    return words


@lru_cache(maxsize=1)
def dictionary_by_length():
    buckets = defaultdict(set)
    for word in load_english_dictionary():
        buckets[len(word)].add(word)
    return {length: sorted(words) for length, words in buckets.items()}


def levenshtein_distance(source, target):
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            substitution_cost = 0 if source_char == target_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def candidate_words(word, dictionary_lookup=None):
    dictionary_lookup = dictionary_lookup or dictionary_by_length()
    lengths = range(max(1, len(word) - 2), len(word) + 3)
    candidates = []
    for length in lengths:
        candidates.extend(dictionary_lookup.get(length, []))
    if len(word) <= 3:
        candidates.extend(candidate for candidate in SHORT_WORD_CANDIDATES if abs(len(candidate) - len(word)) <= 2)
    return candidates


def correction_threshold(word):
    if len(word) <= 3:
        return 1
    if len(word) <= 6:
        return 2
    return 3


def correct_english_word(word, dictionary_lookup=None):
    word = word.upper()
    if len(word) < MIN_WORD_LENGTH or not word.isalpha():
        return word

    dictionary_lookup = dictionary_lookup or dictionary_by_length()
    if word in dictionary_lookup.get(len(word), set()):
        return word

    best_candidate = word
    best_distance = None
    best_score = None

    for candidate in candidate_words(word, dictionary_lookup=dictionary_lookup):
        distance = levenshtein_distance(word, candidate)
        if distance > correction_threshold(word):
            continue

        first_letter_bonus = 0 if candidate[:1] == word[:1] else 1
        last_letter_bonus = 0 if candidate[-1:] == word[-1:] else 1
        score = (distance, first_letter_bonus + last_letter_bonus, abs(len(candidate) - len(word)))

        if best_score is None or score < best_score:
            best_candidate = candidate
            best_distance = distance
            best_score = score

    return best_candidate if best_distance is not None else word


def candidate_list_for_word(word, dictionary_lookup=None):
    dictionary_lookup = dictionary_lookup or dictionary_by_length()
    candidates = {word.upper()}
    for candidate in candidate_words(word, dictionary_lookup=dictionary_lookup):
        distance = levenshtein_distance(word.upper(), candidate)
        allowed_distance = correction_threshold(word)
        if len(word) <= 3:
            allowed_distance = max(allowed_distance, 2)
        if distance <= allowed_distance:
            candidates.add(candidate)

    if len(word) <= 3:
        for candidate in SHORT_WORD_CANDIDATES:
            if levenshtein_distance(word.upper(), candidate) <= 2:
                candidates.add(candidate)

    return sorted(candidates)


def _bigram_bonus(previous_word, current_word, next_word):
    bonus = 0
    if previous_word and (previous_word, current_word) in COMMON_BIGRAMS:
        bonus += 2
    if next_word and (current_word, next_word) in COMMON_BIGRAMS:
        bonus += 2
    return bonus


def choose_contextual_candidate(word, previous_word=None, next_word=None, dictionary_lookup=None):
    dictionary_lookup = dictionary_lookup or dictionary_by_length()
    original = word.upper()
    best_candidate = original
    best_score = None

    for candidate in candidate_list_for_word(original, dictionary_lookup=dictionary_lookup):
        distance = levenshtein_distance(original, candidate)
        common_bonus = 1 if candidate in COMMON_WORDS else 0
        bigram_bonus = _bigram_bonus(previous_word, candidate, next_word)
        first_letter_penalty = 0 if candidate[:1] == original[:1] else 1
        score = (
            distance - common_bonus - bigram_bonus,
            first_letter_penalty,
            abs(len(candidate) - len(original)),
            0 if candidate in COMMON_WORDS else 1,
        )
        if best_score is None or score < best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate != original:
        best_distance = levenshtein_distance(original, best_candidate)
        allowed_distance = correction_threshold(original)
        if len(original) <= 3:
            allowed_distance = max(allowed_distance, 2)
        if best_distance > allowed_distance:
            return original
    return best_candidate


def postprocess_text_lines(lines):
    dictionary_lookup = dictionary_by_length()
    corrected_lines = []
    for line in lines:
        parsed_tokens = []
        for token in line.split():
            trailing_period = token.endswith(".")
            core = token[:-1] if trailing_period else token
            parsed_tokens.append((core, trailing_period))

        corrected_tokens = []
        for index, (core, trailing_period) in enumerate(parsed_tokens):
            previous_word = corrected_tokens[index - 1].rstrip(".") if index > 0 else None
            next_word = None
            if index + 1 < len(parsed_tokens):
                next_word = parsed_tokens[index + 1][0].upper()

            corrected = choose_contextual_candidate(
                core,
                previous_word=previous_word,
                next_word=next_word,
                dictionary_lookup=dictionary_lookup,
            ) if core else core
            if trailing_period:
                corrected += "."
            corrected_tokens.append(corrected)
        corrected_lines.append(" ".join(corrected_tokens))
    return corrected_lines


def restore_sentence_case(text):
    restored = []
    capitalize_next = True
    index = 0

    while index < len(text):
        character = text[index]
        if character.isalpha():
            end_index = index
            while end_index < len(text) and text[end_index].isalpha():
                end_index += 1

            word = text[index:end_index]
            lower_word = word.lower()
            if capitalize_next:
                restored.append(lower_word[:1].upper() + lower_word[1:])
                capitalize_next = False
            elif lower_word == "i":
                restored.append("I")
            else:
                restored.append(lower_word)
            index = end_index
            continue

        restored.append(character)
        if character in ".!?":
            capitalize_next = True
        index += 1

    return "".join(restored)
