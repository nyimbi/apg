"""
NLPC Domain Calculations

All formulas and numeric computations specific to the NLP Core capability.
Every function is pure and type-safe with full edge-case handling.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Confidence / scoring
# ---------------------------------------------------------------------------

def calculate_weighted_confidence(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
	"""
	Compute a weighted average confidence from a label→score mapping.

	If no weights are given, equal weighting is assumed.
	Scores outside [0, 1] are clamped before averaging.

	Returns 0.0 if scores is empty.
	"""
	if not scores:
		return 0.0
	labels = list(scores.keys())
	w = weights or {k: 1.0 for k in labels}
	total_weight = sum(w.get(k, 1.0) for k in labels)
	if total_weight == 0.0:
		return 0.0
	weighted_sum = sum(min(1.0, max(0.0, scores[k])) * w.get(k, 1.0) for k in labels)
	return weighted_sum / total_weight


def calculate_ensemble_confidence(confidences: list[float]) -> float:
	"""
	Combine per-model confidence scores using the harmonic mean.

	Harmonic mean penalises low-confidence outliers more than arithmetic mean,
	providing a conservative ensemble estimate.
	"""
	if not confidences:
		return 0.0
	clamped = [min(1.0, max(1e-9, c)) for c in confidences]
	return len(clamped) / sum(1.0 / c for c in clamped)


def calculate_agreement_score(labels: list[str]) -> float:
	"""
	Inter-annotator agreement as Fleiss-kappa-simplified majority fraction.

	Returns the proportion of annotators who chose the majority label.
	Returns 0.0 if labels is empty.
	"""
	if not labels:
		return 0.0
	from collections import Counter
	counts = Counter(labels)
	majority_count = counts.most_common(1)[0][1]
	return majority_count / len(labels)


# ---------------------------------------------------------------------------
# Sentiment calculations
# ---------------------------------------------------------------------------

def calculate_compound_sentiment(pos: float, neg: float, neu: float) -> float:
	"""
	Compute a VADER-style compound score in [-1, 1].

	Formula: (pos - neg) / (pos + neg + neu + epsilon)
	Neutral mass dampens extreme swings.
	"""
	epsilon = 1e-9
	total = pos + neg + neu + epsilon
	return (pos - neg) / total


def calculate_sentiment_intensity(pos: float, neg: float) -> float:
	"""
	Intensity of sentiment signal regardless of polarity, range [0, 1].

	Defined as max(pos, neg) — the stronger of the two poles.
	"""
	return max(min(1.0, pos), min(1.0, neg))


def normalise_sentiment_scores(pos: float, neg: float, neu: float) -> tuple[float, float, float]:
	"""
	Renormalise three raw sentiment scores so they sum to 1.0.

	Returns (pos, neg, neu) each in [0, 1] summing to 1.0.
	If all three are zero, returns (0.0, 0.0, 1.0) — default neutral.
	"""
	total = pos + neg + neu
	if total <= 0.0:
		return 0.0, 0.0, 1.0
	return pos / total, neg / total, neu / total


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

def calculate_word_count(text: str) -> int:
	"""Approximate word count by whitespace splitting.  O(n) in text length."""
	return len(text.split())


def calculate_sentence_count(text: str) -> int:
	"""
	Approximate sentence count using terminal punctuation heuristic.

	Counts occurrences of '.', '!', '?' not preceded by spaces
	(avoids counting ellipsis fragments).
	"""
	if not text:
		return 0
	import re
	return max(1, len(re.findall(r'[.!?]+', text)))


def calculate_compression_ratio(original_len: int, summary_len: int) -> float:
	"""
	Compression ratio = summary_len / original_len.

	Returns 0.0 if original_len is 0.  Clamps to [0.0, 1.0].
	"""
	if original_len <= 0:
		return 0.0
	return min(1.0, summary_len / original_len)


def calculate_lexical_diversity(text: str) -> float:
	"""
	Type-Token Ratio (TTR): unique tokens / total tokens.

	Simple but interpretable richness measure.  Returns 0.0 for empty text.
	"""
	tokens = text.lower().split()
	if not tokens:
		return 0.0
	return len(set(tokens)) / len(tokens)


def calculate_average_word_length(text: str) -> float:
	"""Return the mean character length of whitespace-separated tokens."""
	words = text.split()
	if not words:
		return 0.0
	return sum(len(w) for w in words) / len(words)


def calculate_reading_difficulty(text: str) -> float:
	"""
	Simplified Flesch-Kincaid readability score (0–100, higher = easier).

	Uses word count, sentence count, and syllable estimation.
	"""
	words = text.split()
	word_count = len(words)
	if word_count == 0:
		return 0.0
	sentence_count = calculate_sentence_count(text)
	syllable_count = sum(_estimate_syllables(w) for w in words)
	# Flesch Reading Ease
	return 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)


def _estimate_syllables(word: str) -> int:
	"""Rough syllable count: count vowel groups."""
	import re
	word = word.lower().strip(".,!?;:")
	if not word:
		return 0
	count = len(re.findall(r'[aeiouy]+', word))
	if word.endswith('e') and count > 1:
		count -= 1
	return max(1, count)


# ---------------------------------------------------------------------------
# Embedding / similarity
# ---------------------------------------------------------------------------

def calculate_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
	"""
	Cosine similarity between two vectors.  Returns 0.0 for zero vectors.

	Raises ValueError if vectors have different lengths.
	"""
	if len(vec_a) != len(vec_b):
		raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")
	dot = sum(a * b for a, b in zip(vec_a, vec_b))
	norm_a = math.sqrt(sum(a * a for a in vec_a))
	norm_b = math.sqrt(sum(b * b for b in vec_b))
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (norm_a * norm_b)


def calculate_l2_norm(vector: list[float]) -> float:
	"""Euclidean (L2) norm of a vector."""
	return math.sqrt(sum(x * x for x in vector))


def normalise_vector(vector: list[float]) -> list[float]:
	"""Return unit-length version of vector.  Returns zero vector unchanged."""
	norm = calculate_l2_norm(vector)
	if norm == 0.0:
		return vector[:]
	return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# TF-IDF keyword scoring
# ---------------------------------------------------------------------------

def calculate_tf(term: str, document: str) -> float:
	"""Term frequency: count of term occurrences / total tokens."""
	tokens = document.lower().split()
	if not tokens:
		return 0.0
	term_lower = term.lower()
	return tokens.count(term_lower) / len(tokens)


def calculate_idf(term: str, corpus: list[str]) -> float:
	"""
	Inverse document frequency (smoothed): log((1 + N) / (1 + df)) + 1.

	Returns 0.0 for an empty corpus.
	"""
	n = len(corpus)
	if n == 0:
		return 0.0
	term_lower = term.lower()
	df = sum(1 for doc in corpus if term_lower in doc.lower())
	return math.log((1.0 + n) / (1.0 + df)) + 1.0


def calculate_tfidf(term: str, document: str, corpus: list[str]) -> float:
	"""TF-IDF score = TF * IDF."""
	return calculate_tf(term, document) * calculate_idf(term, corpus)


# ---------------------------------------------------------------------------
# Language detection quality
# ---------------------------------------------------------------------------

def calculate_language_certainty(probabilities: list[float]) -> float:
	"""
	Certainty of language detection as 1 − entropy(probs) / log(N).

	Returns 1.0 (perfectly certain) when one language dominates.
	Returns 0.0 when uniform uncertainty across all candidates.
	"""
	probs = [max(0.0, p) for p in probabilities]
	total = sum(probs)
	if total == 0.0 or len(probs) <= 1:
		return 1.0
	normed = [p / total for p in probs]
	entropy = -sum(p * math.log(p) for p in normed if p > 0)
	max_entropy = math.log(len(normed))
	return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0


# ---------------------------------------------------------------------------
# Batch / throughput
# ---------------------------------------------------------------------------

def calculate_throughput(doc_count: int, elapsed_seconds: float) -> float:
	"""Documents per second. Returns 0.0 for zero elapsed time."""
	if elapsed_seconds <= 0:
		return 0.0
	return doc_count / elapsed_seconds


def calculate_estimated_completion_seconds(
	total: int,
	processed: int,
	elapsed_seconds: float,
) -> float | None:
	"""
	Estimate remaining time for a batch job.

	Returns None if no documents have been processed yet.
	"""
	if processed <= 0 or elapsed_seconds <= 0:
		return None
	rate = processed / elapsed_seconds
	remaining = total - processed
	return remaining / rate if rate > 0 else None


def calculate_error_rate(failed: int, total: int) -> float:
	"""Error rate in [0.0, 1.0]. Returns 0.0 for zero total."""
	if total <= 0:
		return 0.0
	return min(1.0, failed / total)


def calculate_cache_hit_rate(hits: int, requests: int) -> float:
	"""Cache hit rate in [0.0, 1.0]. Returns 0.0 for zero requests."""
	if requests <= 0:
		return 0.0
	return min(1.0, hits / requests)


# ---------------------------------------------------------------------------
# Percentile (for p95 latency)
# ---------------------------------------------------------------------------

def calculate_percentile(values: Sequence[float], percentile: float) -> float:
	"""
	Compute the p-th percentile of a sequence using linear interpolation.

	percentile must be in [0, 100].  Returns 0.0 for empty sequences.
	"""
	if not values:
		return 0.0
	if not 0.0 <= percentile <= 100.0:
		raise ValueError(f"percentile must be in [0, 100], got {percentile}")
	sorted_vals = sorted(values)
	n = len(sorted_vals)
	index = (percentile / 100.0) * (n - 1)
	lower = int(index)
	upper = lower + 1
	if upper >= n:
		return float(sorted_vals[-1])
	frac = index - lower
	return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
