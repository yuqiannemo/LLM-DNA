#!/usr/bin/env python3
"""
Generate classical Chinese (文言文) random sentence dataset for DNA extraction.

Creates a dataset of 100 samples, each containing ~100 characters of
grammatically valid classical Chinese sentences using a context-free grammar
with curated vocabulary from classical_vocab.json.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_VOCAB_FILE = Path(__file__).resolve().parent / "classical_vocab.json"

# Context-free grammar for classical Chinese (文言文).
# Function words are fixed in the grammar; content words reference vocab categories.
_GRAMMAR = {
    "S": [
        "{CLAUSE}。",
        "{CLAUSE}，{CLAUSE}。",
        "{JUDGMENT}。",
        "{CONDITIONAL}。",
        "{CONCESSIVE}。",
        "{SEQUENCE}。",
        "{PARALLEL}。",
    ],
    "CLAUSE": [
        "{SUBJ_OPT}{VP}",
        "{SUBJ_OPT}{VP}于{_PLACE}",
        "{SUBJ_OPT}{VP}而{VP}",
        "以{_ABSTRACT_NOUN}为{NP}",
    ],
    "CLAUSE_SHORT": [
        "{SUBJ_OPT}{VP}",
        "{SUBJ_OPT}{VP}于{_PLACE}",
        "以{_ABSTRACT_NOUN}为{NP}",
    ],
    "JUDGMENT": [
        "{NP}者，{PRED}也",
        "{_HUMAN_NOUN}者，{VP}者也",
        "{_ABSTRACT_NOUN}者，{_ABSTRACT_NOUN}之本也",
    ],
    "PRED": [
        "{_ADJ}",
        "{_ABSTRACT_NOUN}之{_ADJ}者",
        "{_THING_NOUN}之{_ADJ}者",
    ],
    "CONDITIONAL": [
        "若{CLAUSE_SHORT}，则{CLAUSE_SHORT}",
        "苟{CLAUSE_SHORT}，则{CLAUSE_SHORT}",
    ],
    "CONCESSIVE": [
        "虽{CLAUSE_SHORT}，犹{CLAUSE_SHORT}",
        "虽{CLAUSE_SHORT}，而{CLAUSE_SHORT}",
    ],
    "SEQUENCE": [
        "既{CLAUSE_SHORT}，乃{CLAUSE_SHORT}",
    ],
    "PARALLEL": [
        "{FOURCHAR}，{FOURCHAR}",
        "{PAIR_VP}，{PAIR_VP}",
        "或{VP}，或{VP}",
    ],
    "FOURCHAR": [
        "{_MONO_VERB}{_MONO_NOUN}{_MONO_VERB}{_MONO_NOUN}",
        "{_MONO_ADJ}{_MONO_NOUN}{_MONO_ADJ}{_MONO_NOUN}",
    ],
    "PAIR_VP": [
        "{_VERB_MORAL}{_ABSTRACT_NOUN}而{_VERB_MORAL}{_ABSTRACT_NOUN}",
        "{_VERB_MENTAL}{_THING_NOUN}而{_VERB_MENTAL}{_ABSTRACT_NOUN}",
    ],
    "VP": [
        "{_VERB_MORAL}{OBJ}",
        "{_VERB_MENTAL}{_ABSTRACT_NOUN}",
        "{_VERB_GOVERN}{_STATE_NOUN}",
        "{_VERB_MOTION}{_PLACE}",
        "{_VERB_CHANGE}",
        "不{_VERB_MORAL}{OBJ}",
        "未{_VERB_MENTAL}{_ABSTRACT_NOUN}",
    ],
    "OBJ": [
        "{_ABSTRACT_NOUN}",
        "{_THING_NOUN}",
        "{_STATE_NOUN}",
    ],
    "NP": [
        "{_HUMAN_NOUN}",
        "{_ABSTRACT_NOUN}",
        "{_THING_NOUN}",
        "{_STATE_NOUN}",
        "{_MONO_NOUN}之{_MONO_NOUN}",
        "{_ADJ}{_MONO_NOUN}",
        "{_ADJ}之{_MONO_NOUN}",
    ],
    "SUBJ_OPT": [
        "",
        "",
        "{_HUMAN_NOUN}",
        "{_THING_NOUN}",
        "{_STATE_NOUN}",
    ],
}

_MAX_DEPTH = 6

# Mapping from grammar terminal names to vocab keys
_TERMINAL_MAP = {
    "_HUMAN_NOUN": "human_nouns",
    "_ABSTRACT_NOUN": "abstract_nouns",
    "_THING_NOUN": "thing_nouns",
    "_STATE_NOUN": "state_nouns",
    "_PLACE": "places",
    "_VERB_MORAL": "verbs_moral",
    "_VERB_MENTAL": "verbs_mental",
    "_VERB_GOVERN": "verbs_govern",
    "_VERB_MOTION": "verbs_motion",
    "_VERB_CHANGE": "verbs_change",
    "_ADJ": "adjectives",
    "_MONO_NOUN": "mono_nouns",
    "_MONO_VERB": "mono_verbs",
    "_MONO_ADJ": "mono_adjectives",
    "_DUAL": "dual_terms",
}


def _load_vocab(path: Path = _VOCAB_FILE) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expand(symbol: str, vocab: Dict[str, List[str]], depth: int = 0) -> str:
    """Recursively expand a grammar symbol into a string."""
    if depth > _MAX_DEPTH:
        return random.choice(vocab["mono_nouns"])

    # Terminal: pick a random word from the matching vocab category
    if symbol in _TERMINAL_MAP:
        return random.choice(vocab[_TERMINAL_MAP[symbol]])

    # Non-terminal: pick a random production and expand
    if symbol in _GRAMMAR:
        production = random.choice(_GRAMMAR[symbol])
        result = production
        while "{" in result:
            start = result.index("{")
            end = result.index("}", start)
            token = result[start + 1 : end]
            expanded = _expand(token, vocab, depth + 1)
            result = result[:start] + expanded + result[end + 1 :]
        return result

    return symbol


def _generate_sentence(vocab: Dict[str, List[str]]) -> str:
    """Generate a single classical Chinese sentence from the CFG."""
    return _expand("S", vocab)


def _count_chars(text: str) -> int:
    """Count meaningful characters (exclude punctuation)."""
    return sum(1 for c in text if c not in "，。、；：！？")


def generate_random_chinese_samples(
    num_samples: int = 100,
    chars_per_sample: int = 100,
    seed: int = 42,
    vocab_path: Path = _VOCAB_FILE,
) -> List[str]:
    """
    Generate samples of classical Chinese sentences from a CFG.

    Args:
        num_samples: Number of samples to generate
        chars_per_sample: Target number of characters per sample
        seed: Random seed for reproducibility
        vocab_path: Path to the vocabulary JSON file

    Returns:
        List of strings, each containing ~chars_per_sample characters
    """
    random.seed(seed)
    vocab = _load_vocab(vocab_path)
    samples = []

    logger.info(
        f"Generating {num_samples} classical Chinese samples "
        f"with ~{chars_per_sample} chars each..."
    )

    for i in range(num_samples):
        sentences: List[str] = []
        char_count = 0
        while char_count < chars_per_sample:
            sent = _generate_sentence(vocab)
            sentences.append(sent)
            char_count += _count_chars(sent)
        sample = "".join(sentences)
        samples.append(sample)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")

    logger.info(f"Successfully generated {len(samples)} samples")
    return samples


def save_dataset(
    samples: List[str],
    output_file: Path,
) -> None:
    """Save dataset to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(samples)} samples to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate classical Chinese (文言文) random dataset for DNA extraction"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--chars-per-sample",
        type=int,
        default=100,
        help="Target characters per sample (default: 100)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/rand/rand_dataset_chinese.json"),
        help="Output file path (default: data/rand/rand_dataset_chinese.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=_VOCAB_FILE,
        help="Path to vocabulary JSON file",
    )

    args = parser.parse_args()

    samples = generate_random_chinese_samples(
        num_samples=args.num_samples,
        chars_per_sample=args.chars_per_sample,
        seed=args.seed,
        vocab_path=args.vocab_file,
    )

    save_dataset(samples, args.output_file)
    logger.info("Dataset generation completed successfully!")


if __name__ == "__main__":
    main()
