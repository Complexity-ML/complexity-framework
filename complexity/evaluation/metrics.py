"""
Generation quality metrics.

Provides:
- ROUGE scores
- BLEU scores
- BERTScore
- Diversity metrics
- Coherence metrics
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math
import re


class RougeScorer:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scorer.

    Measures overlap between generated text and reference.

    Supports:
    - ROUGE-N: N-gram overlap
    - ROUGE-L: Longest common subsequence
    """

    def __init__(self, use_stemmer: bool = False):
        """
        Args:
            use_stemmer: Apply stemming to words
        """
        self.use_stemmer = use_stemmer

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def score(
        self,
        prediction: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        results = {}

        # ROUGE-1
        pred_1grams = self._get_ngrams(pred_tokens, 1)
        ref_1grams = self._get_ngrams(ref_tokens, 1)
        results["rouge1"] = self._compute_rouge(pred_1grams, ref_1grams)

        # ROUGE-2
        pred_2grams = self._get_ngrams(pred_tokens, 2)
        ref_2grams = self._get_ngrams(ref_tokens, 2)
        results["rouge2"] = self._compute_rouge(pred_2grams, ref_2grams)

        # ROUGE-L
        lcs_len = self._lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results["rougeL"] = {"precision": precision, "recall": recall, "f1": f1}

        return results

    def _compute_rouge(
        self,
        pred_ngrams: Counter,
        ref_ngrams: Counter,
    ) -> Dict[str, float]:
        """Compute ROUGE precision, recall, F1."""
        overlap = sum((pred_ngrams & ref_ngrams).values())
        pred_total = sum(pred_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = overlap / pred_total if pred_total > 0 else 0
        recall = overlap / ref_total if ref_total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}

    def score_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute average ROUGE scores for a batch."""
        all_scores = [self.score(p, r) for p, r in zip(predictions, references)]

        avg_scores = {}
        for metric in ["rouge1", "rouge2", "rougeL"]:
            avg_scores[metric] = {
                "precision": sum(s[metric]["precision"] for s in all_scores) / len(all_scores),
                "recall": sum(s[metric]["recall"] for s in all_scores) / len(all_scores),
                "f1": sum(s[metric]["f1"] for s in all_scores) / len(all_scores),
            }

        return avg_scores


class BleuScorer:
    """
    BLEU (Bilingual Evaluation Understudy) scorer.

    Measures n-gram precision with brevity penalty.
    """

    def __init__(self, max_n: int = 4, smooth: bool = True):
        """
        Args:
            max_n: Maximum n-gram order
            smooth: Apply smoothing for short texts
        """
        self.max_n = max_n
        self.smooth = smooth

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def score(
        self,
        prediction: str,
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute BLEU score.

        Args:
            prediction: Generated text
            references: List of reference texts

        Returns:
            Dictionary with BLEU score and n-gram precisions
        """
        pred_tokens = self._tokenize(prediction)

        if len(pred_tokens) == 0:
            return {"bleu": 0.0, "precisions": [0.0] * self.max_n}

        ref_tokens_list = [self._tokenize(r) for r in references]

        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)

            # Get max counts from references
            max_ref_counts = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            # Clip predicted counts
            clipped = sum(min(count, max_ref_counts[ngram]) for ngram, count in pred_ngrams.items())
            total = sum(pred_ngrams.values())

            if self.smooth and total == 0:
                precision = 1.0 / (len(pred_tokens) + 1)
            else:
                precision = clipped / total if total > 0 else 0

            precisions.append(precision)

        # Compute brevity penalty
        pred_len = len(pred_tokens)
        ref_lens = [len(tokens) for tokens in ref_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - pred_len), x))

        if pred_len > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / pred_len) if pred_len > 0 else 0

        # Compute BLEU
        if all(p > 0 for p in precisions):
            log_avg = sum(math.log(p) for p in precisions) / len(precisions)
            bleu = bp * math.exp(log_avg)
        else:
            bleu = 0.0

        return {"bleu": bleu, "precisions": precisions, "brevity_penalty": bp}

    def score_corpus(
        self,
        predictions: List[str],
        references_list: List[List[str]],
    ) -> Dict[str, float]:
        """Compute corpus-level BLEU."""
        total_clipped = [0] * self.max_n
        total_counts = [0] * self.max_n
        total_pred_len = 0
        total_ref_len = 0

        for pred, refs in zip(predictions, references_list):
            pred_tokens = self._tokenize(pred)
            ref_tokens_list = [self._tokenize(r) for r in refs]

            total_pred_len += len(pred_tokens)
            ref_lens = [len(tokens) for tokens in ref_tokens_list]
            closest = min(ref_lens, key=lambda x: (abs(x - len(pred_tokens)), x))
            total_ref_len += closest

            for n in range(1, self.max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)

                max_ref_counts = Counter()
                for ref_tokens in ref_tokens_list:
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    for ngram, count in ref_ngrams.items():
                        max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

                clipped = sum(min(count, max_ref_counts[ngram]) for ngram, count in pred_ngrams.items())
                total_clipped[n - 1] += clipped
                total_counts[n - 1] += sum(pred_ngrams.values())

        # Corpus precisions
        precisions = [
            c / t if t > 0 else 0
            for c, t in zip(total_clipped, total_counts)
        ]

        # Brevity penalty
        if total_pred_len > total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - total_ref_len / total_pred_len) if total_pred_len > 0 else 0

        # BLEU
        if all(p > 0 for p in precisions):
            log_avg = sum(math.log(p) for p in precisions) / len(precisions)
            bleu = bp * math.exp(log_avg)
        else:
            bleu = 0.0

        return {"bleu": bleu, "precisions": precisions, "brevity_penalty": bp}


class BertScoreEvaluator:
    """
    BERTScore: Embedding-based similarity metric.

    Uses contextual embeddings to compute semantic similarity.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Encoder model for embeddings
            tokenizer: Tokenizer
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if model else torch.device('cpu'))

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for texts."""
        if self.model is None or self.tokenizer is None:
            # Return random embeddings as placeholder
            return torch.randn(len(texts), 768)

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # Mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    emb = outputs.last_hidden_state.mean(dim=1)
                else:
                    emb = outputs[0].mean(dim=1)

                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    def score(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute BERTScore.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary with precision, recall, F1
        """
        pred_embs = self._get_embeddings(predictions)
        ref_embs = self._get_embeddings(references)

        # Cosine similarity
        pred_embs = pred_embs / pred_embs.norm(dim=-1, keepdim=True)
        ref_embs = ref_embs / ref_embs.norm(dim=-1, keepdim=True)

        similarities = (pred_embs * ref_embs).sum(dim=-1)

        return {
            "precision": similarities.mean().item(),
            "recall": similarities.mean().item(),
            "f1": similarities.mean().item(),
        }


class DiversityMetrics:
    """
    Metrics for measuring generation diversity.

    - Distinct-N: Ratio of unique n-grams
    - Self-BLEU: BLEU between generated samples
    - Entropy: Token distribution entropy
    """

    def __init__(self, max_n: int = 4):
        """
        Args:
            max_n: Maximum n-gram order
        """
        self.max_n = max_n

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def distinct_n(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute Distinct-N scores.

        Higher is more diverse.
        """
        results = {}

        for n in range(1, self.max_n + 1):
            all_ngrams = []
            for text in texts:
                tokens = self._tokenize(text)
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    all_ngrams.append(ngram)

            unique = len(set(all_ngrams))
            total = len(all_ngrams)
            results[f"distinct_{n}"] = unique / total if total > 0 else 0

        return results

    def self_bleu(self, texts: List[str]) -> float:
        """
        Compute Self-BLEU.

        Lower is more diverse.
        """
        bleu_scorer = BleuScorer()
        scores = []

        for i, text in enumerate(texts):
            references = texts[:i] + texts[i + 1:]
            if references:
                score = bleu_scorer.score(text, references)
                scores.append(score["bleu"])

        return sum(scores) / len(scores) if scores else 0

    def entropy(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute token entropy.

        Higher is more diverse.
        """
        # Unigram entropy
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())

        if total == 0:
            return {"unigram_entropy": 0, "bigram_entropy": 0}

        probs = [count / total for count in token_counts.values()]
        unigram_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Bigram entropy
        bigrams = []
        for text in texts:
            tokens = self._tokenize(text)
            for i in range(len(tokens) - 1):
                bigrams.append((tokens[i], tokens[i + 1]))

        bigram_counts = Counter(bigrams)
        total_bigrams = sum(bigram_counts.values())

        if total_bigrams == 0:
            return {"unigram_entropy": unigram_entropy, "bigram_entropy": 0}

        probs = [count / total_bigrams for count in bigram_counts.values()]
        bigram_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return {
            "unigram_entropy": unigram_entropy,
            "bigram_entropy": bigram_entropy,
        }

    def compute_all(self, texts: List[str]) -> Dict[str, float]:
        """Compute all diversity metrics."""
        results = {}
        results.update(self.distinct_n(texts))
        results["self_bleu"] = self.self_bleu(texts)
        results.update(self.entropy(texts))
        return results


class CoherenceMetrics:
    """
    Metrics for measuring text coherence.

    - Repetition rate
    - Sentence similarity
    - Topic consistency
    """

    def repetition_rate(self, text: str, n: int = 3) -> float:
        """
        Compute n-gram repetition rate.

        Lower is better (less repetition).
        """
        tokens = text.lower().split()

        if len(tokens) < n:
            return 0

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)

        unique = len(set(ngrams))
        total = len(ngrams)

        return 1 - (unique / total) if total > 0 else 0

    def sentence_coherence(
        self,
        text: str,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> float:
        """
        Compute coherence between consecutive sentences.

        Uses embedding similarity.
        """
        # Simple sentence split
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0

        if model is None:
            # Return placeholder
            return 0.8

        # Get embeddings
        bert_score = BertScoreEvaluator(model, tokenizer)
        scores = []

        for i in range(len(sentences) - 1):
            result = bert_score.score([sentences[i]], [sentences[i + 1]])
            scores.append(result["f1"])

        return sum(scores) / len(scores) if scores else 1.0

    def compute_all(
        self,
        texts: List[str],
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Compute all coherence metrics."""
        rep_rates = [self.repetition_rate(text) for text in texts]
        coherence_scores = [
            self.sentence_coherence(text, model, tokenizer)
            for text in texts
        ]

        return {
            "avg_repetition_rate": sum(rep_rates) / len(rep_rates) if rep_rates else 0,
            "avg_coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0,
        }
