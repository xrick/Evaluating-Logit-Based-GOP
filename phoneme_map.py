# This file defines the Phoneme Confusion Map based on Section 2.3.1 of the paper.
# It includes examples of phonetic proximity, common L2 learner errors, and phonological rules.

PHONEME_CONFUSION_MAP = {
    # Common L2 learner errors (examples from the paper [cite: 63, 125])
    'θ': ['s', 't', 'f'],  # dental fricative substitutions
    'ð': ['d', 'z', 'v'],
    'æ': ['ɛ', 'e', 'a'], # /æ/ vs /e/
    'ʌ': ['a', 'ə'],

    # Phonetic Proximity (e.g., stops with similar place of articulation)
    'p': ['b', 'm'],
    't': ['d', 'n', 'ɾ'], # 'ɾ' is a flap, an allophone of /t/ or /d/
    'k': ['g', 'ŋ'],
    'b': ['p', 'm'],
    'd': ['t', 'n', 'ɾ'],
    'g': ['k', 'ŋ'],

    # Vowel Mergers / Diphthong Simplification
    'eɪ': ['e', 'ɛ'], # e.g., /eɪ/ -> /e:/
    'oʊ': ['o', 'ɔ'], # e.g., /oʊ/ -> /o:/
    'aɪ': ['a', 'i'],
    'ɔɪ': ['ɔ', 'i'],
    'aʊ': ['a', 'u'],

    # Other common confusions
    'l': ['ɹ', 'w'],
    'ɹ': ['l', 'w'],
    'v': ['w', 'f'],
    'ʃ': ['s', 'tʃ'],
    'z': ['s'],
}

def get_restricted_substitutions(phoneme: str) -> list:
    """
    Returns the list of allowed substitutions for a given phoneme.
    """
    return PHONEME_CONFUSION_MAP.get(phoneme, [])