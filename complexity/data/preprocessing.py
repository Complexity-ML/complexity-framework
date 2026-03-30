"""
Data preprocessing utilities.

Provides:
- Text cleaning and normalization
- Data augmentation
- Sequence packing
- Efficient batching
"""

import torch
from typing import List, Dict, Any, Optional, Callable
import re
import random


class TextPreprocessor:
    """
    Text preprocessing pipeline.

    Handles:
    - Unicode normalization
    - Whitespace cleaning
    - Special character handling
    - Language filtering
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        remove_html: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None,
    ):
        """
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_whitespace: Normalize whitespace
            remove_html: Remove HTML tags
            min_length: Minimum text length (characters)
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_html = remove_html
        self.min_length = min_length
        self.max_length = max_length

        # Compile regexes
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+'
        )
        self.email_pattern = re.compile(
            r'[\w\.-]+@[\w\.-]+\.\w+'
        )
        self.html_pattern = re.compile(
            r'<[^>]+>'
        )
        self.whitespace_pattern = re.compile(
            r'\s+'
        )

    def __call__(self, text: str) -> Optional[str]:
        """
        Preprocess text.

        Args:
            text: Input text

        Returns:
            Preprocessed text or None if filtered out
        """
        if not text:
            return None

        # Remove HTML
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)

        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Length filtering
        if len(text) < self.min_length:
            return None

        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text


class DataPipeline:
    """
    Composable data preprocessing pipeline.

    Chains multiple preprocessing steps together.
    """

    def __init__(self, steps: List[Callable[[Any], Any]]):
        """
        Args:
            steps: List of preprocessing functions
        """
        self.steps = steps

    def __call__(self, data: Any) -> Optional[Any]:
        """Apply pipeline steps."""
        for step in self.steps:
            if data is None:
                return None
            data = step(data)
        return data

    def add_step(self, step: Callable[[Any], Any]):
        """Add a preprocessing step."""
        self.steps.append(step)

    @classmethod
    def create_default(cls) -> "DataPipeline":
        """Create default text preprocessing pipeline."""
        preprocessor = TextPreprocessor()
        return cls([preprocessor])


class PackedSequenceCollator:
    """
    Collator that packs multiple sequences into single examples.

    Maximizes GPU utilization by avoiding padding waste.

    Example:
        # Without packing (lots of padding):
        [[1,2,3,PAD,PAD], [1,2,3,4,5], [1,2,PAD,PAD,PAD]]

        # With packing (no padding waste):
        [[1,2,3,EOS,1,2,3,4,5,EOS,1,2]]
    """

    def __init__(
        self,
        seq_length: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        pack_sequences: bool = True,
    ):
        """
        Args:
            seq_length: Target sequence length
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            pack_sequences: Whether to pack sequences
        """
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pack_sequences = pack_sequences

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with optional packing.

        Args:
            batch: List of examples with 'input_ids'

        Returns:
            Batched and optionally packed tensors
        """
        if self.pack_sequences:
            return self._pack_batch(batch)
        else:
            return self._pad_batch(batch)

    def _pad_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Standard padding collation."""
        max_len = min(
            max(item['input_ids'].size(0) for item in batch),
            self.seq_length
        )

        input_ids = torch.full(
            (len(batch), max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = min(item['input_ids'].size(0), max_len)
            input_ids[i, :seq_len] = item['input_ids'][:seq_len]
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = item['input_ids'][:seq_len]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _pack_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pack multiple sequences into single examples."""
        # Collect all tokens
        all_tokens = []
        for item in batch:
            tokens = item['input_ids'].tolist()
            all_tokens.extend(tokens)
            all_tokens.append(self.eos_token_id)

        # Pack into sequences of seq_length
        packed_sequences = []
        i = 0
        while i < len(all_tokens):
            seq = all_tokens[i:i + self.seq_length]
            if len(seq) < self.seq_length:
                # Pad the last sequence
                seq = seq + [self.pad_token_id] * (self.seq_length - len(seq))
            packed_sequences.append(seq)
            i += self.seq_length

        # Create batch
        if not packed_sequences:
            # Fallback to single padded sequence
            return self._pad_batch(batch)

        input_ids = torch.tensor(packed_sequences, dtype=torch.long)

        # Create attention mask (1 for non-pad tokens)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Labels are same as input_ids, with -100 for padding
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class DataAugmenter:
    """
    Data augmentation for text.

    Techniques:
    - Random deletion
    - Random swap
    - Synonym replacement (requires external data)
    - Back-translation (requires models)
    """

    def __init__(
        self,
        delete_prob: float = 0.0,
        swap_prob: float = 0.0,
        insert_prob: float = 0.0,
    ):
        """
        Args:
            delete_prob: Probability of deleting a token
            swap_prob: Probability of swapping adjacent tokens
            insert_prob: Probability of inserting duplicate token
        """
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob
        self.insert_prob = insert_prob

    def augment_tokens(self, tokens: List[int], seed: Optional[int] = None) -> List[int]:
        """
        Augment a sequence of tokens.

        Args:
            tokens: Input token IDs
            seed: Random seed for reproducibility

        Returns:
            Augmented tokens
        """
        if seed is not None:
            random.seed(seed)

        tokens = list(tokens)

        # Random deletion
        if self.delete_prob > 0:
            tokens = [t for t in tokens if random.random() > self.delete_prob]

        # Random swap
        if self.swap_prob > 0:
            for i in range(len(tokens) - 1):
                if random.random() < self.swap_prob:
                    tokens[i], tokens[i+1] = tokens[i+1], tokens[i]

        # Random insertion (duplicate)
        if self.insert_prob > 0:
            new_tokens = []
            for t in tokens:
                new_tokens.append(t)
                if random.random() < self.insert_prob:
                    new_tokens.append(t)
            tokens = new_tokens

        return tokens

    def __call__(self, item: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Augment a data item."""
        tokens = item['input_ids'].tolist()
        augmented = self.augment_tokens(tokens)
        return {
            'input_ids': torch.tensor(augmented, dtype=torch.long),
        }


class MixupCollator:
    """
    Mixup data augmentation for sequence classification.

    Mixes pairs of examples to create soft labels.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixup ratio
        """
        self.alpha = alpha

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Apply mixup to batch."""
        # Standard collation first
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        # Sample mixup ratio
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        # Random permutation for mixing
        perm = torch.randperm(input_ids.size(0))

        # Mix inputs (for embeddings, this would be done in model)
        # Here we just return both for the model to handle
        return {
            'input_ids': input_ids,
            'input_ids_mixed': input_ids[perm],
            'labels': labels,
            'labels_mixed': labels[perm],
            'mixup_lambda': lam,
        }
