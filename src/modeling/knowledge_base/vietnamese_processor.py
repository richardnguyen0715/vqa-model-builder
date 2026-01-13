"""
Vietnamese Text Processor Module
Provides Vietnamese-specific text processing utilities.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


# Vietnamese character patterns
VIETNAMESE_CHARS = (
    'aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơờởỡớợ'
    'uùủũúụưừửữứựyỳỷỹýỵđ'
    'AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬEÈẺẼÉẸÊỀỂỄẾỆIÌỈĨÍỊOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢ'
    'UÙỦŨÚỤƯỪỬỮỨỰYỲỶỸÝỴĐ'
)

# Common Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'là', 'có', 'được', 'trong', 'với', 'này', 'cho', 'một',
    'các', 'những', 'để', 'theo', 'như', 'từ', 'đã', 'cũng', 'khi', 'về',
    'sẽ', 'nếu', 'còn', 'đến', 'hay', 'hoặc', 'bị', 'vì', 'do', 'lại',
    'ra', 'vào', 'sau', 'trước', 'tại', 'đây', 'kia', 'đó', 'nào', 'gì',
    'ai', 'sao', 'thì', 'mà', 'nhưng', 'tuy', 'dù', 'mỗi', 'mọi', 'tất',
    'đều', 'rất', 'lắm', 'quá', 'hơn', 'nhất', 'nữa', 'thêm', 'bao', 'nhiêu'
}

# Tone/diacritics mapping for normalization
TONE_MAPPING = {
    # a variants
    'à': 'a', 'ả': 'a', 'ã': 'a', 'á': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ắ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ấ': 'a', 'ậ': 'a',
    # e variants
    'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'é': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ế': 'e', 'ệ': 'e',
    # i variants
    'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'í': 'i', 'ị': 'i',
    # o variants
    'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ó': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ố': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ớ': 'o', 'ợ': 'o',
    # u variants
    'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ú': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ứ': 'u', 'ự': 'u',
    # y variants
    'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ý': 'y', 'ỵ': 'y',
    # d variant
    'đ': 'd',
}


def normalize_vietnamese_text(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = False,
    remove_punctuation: bool = False,
    normalize_unicode: bool = True
) -> str:
    """
    Normalize Vietnamese text.
    
    Args:
        text: Input text
        lowercase: Whether to lowercase
        remove_accents: Whether to remove diacritics
        remove_punctuation: Whether to remove punctuation
        normalize_unicode: Whether to normalize unicode
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC form)
    if normalize_unicode:
        text = unicodedata.normalize('NFC', text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove accents/diacritics
    if remove_accents:
        result = []
        for char in text:
            lower_char = char.lower()
            if lower_char in TONE_MAPPING:
                result.append(TONE_MAPPING[lower_char] if char.islower() else TONE_MAPPING[lower_char].upper())
            else:
                result.append(char)
        text = ''.join(result)
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class VietnameseTokenizer:
    """
    Vietnamese tokenizer with word segmentation support.
    Uses VnCoreNLP or underthesea for word segmentation.
    """
    
    def __init__(
        self,
        use_word_segmentation: bool = True,
        segmenter: str = 'underthesea',
        keep_special_tokens: bool = True
    ):
        """
        Initialize Vietnamese tokenizer.
        
        Args:
            use_word_segmentation: Whether to use word segmentation
            segmenter: Segmentation library (underthesea, vncorenlp)
            keep_special_tokens: Whether to keep special tokens
        """
        self.use_word_segmentation = use_word_segmentation
        self.segmenter = segmenter
        self.keep_special_tokens = keep_special_tokens
        
        self._segmenter_func = None
        if use_word_segmentation:
            self._init_segmenter()
    
    def _init_segmenter(self):
        """Initialize word segmentation function."""
        if self.segmenter == 'underthesea':
            try:
                from underthesea import word_tokenize
                self._segmenter_func = word_tokenize
            except ImportError:
                import warnings
                warnings.warn(
                    "underthesea not installed. "
                    "Install with: pip install underthesea"
                )
                self._segmenter_func = None
                
        elif self.segmenter == 'vncorenlp':
            try:
                from vncorenlp import VnCoreNLP
                # Note: Requires VnCoreNLP jar file
                import warnings
                warnings.warn(
                    "VnCoreNLP requires additional setup. "
                    "Falling back to simple tokenization."
                )
                self._segmenter_func = None
            except ImportError:
                self._segmenter_func = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Normalize first
        text = normalize_vietnamese_text(text, lowercase=False)
        
        # Word segmentation
        if self.use_word_segmentation and self._segmenter_func:
            try:
                tokens = self._segmenter_func(text)
                if isinstance(tokens, str):
                    tokens = tokens.split()
                return tokens
            except Exception:
                pass
        
        # Simple whitespace tokenization
        return text.split()
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]


class VietnameseSentenceSplitter:
    """
    Vietnamese sentence splitter.
    Handles Vietnamese-specific sentence boundaries.
    """
    
    # Vietnamese sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[.!?])\s+(?=[A-ZÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬĐEÈẺẼÉẸÊỀỂỄẾỆIÌỈĨÍỊOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢUÙỦŨÚỤƯỪỬỮỨỰYỲỶỸÝỴ])'
    )
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 500
    ):
        """
        Initialize sentence splitter.
        
        Args:
            min_length: Minimum sentence length
            max_length: Maximum sentence length
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Normalize
        text = normalize_vietnamese_text(text, lowercase=False)
        
        # Split by sentence endings
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        # Also split by newlines
        result = []
        for sent in sentences:
            parts = sent.split('\n')
            result.extend(parts)
        
        # Filter by length
        result = [
            s.strip() for s in result
            if self.min_length <= len(s.strip()) <= self.max_length
        ]
        
        return result


class VietnameseTextProcessor:
    """
    Comprehensive Vietnamese text processor.
    Combines normalization, tokenization, and sentence splitting.
    """
    
    def __init__(
        self,
        use_word_segmentation: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        stopwords: Optional[set] = None,
        normalize_accents: bool = False
    ):
        """
        Initialize Vietnamese text processor.
        
        Args:
            use_word_segmentation: Whether to use word segmentation
            lowercase: Whether to lowercase text
            remove_stopwords: Whether to remove stopwords
            stopwords: Custom stopword set
            normalize_accents: Whether to normalize accents
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords or VIETNAMESE_STOPWORDS
        self.normalize_accents = normalize_accents
        
        self.tokenizer = VietnameseTokenizer(
            use_word_segmentation=use_word_segmentation
        )
        self.sentence_splitter = VietnameseSentenceSplitter()
    
    def process(self, text: str) -> str:
        """
        Process Vietnamese text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Normalize
        text = normalize_vietnamese_text(
            text,
            lowercase=self.lowercase,
            remove_accents=self.normalize_accents
        )
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]
        
        return ' '.join(tokens)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of processed texts
        """
        return [self.process(text) for text in texts]
    
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        use_tfidf: bool = False
    ) -> List[str]:
        """
        Extract keywords from Vietnamese text.
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            use_tfidf: Whether to use TF-IDF scoring
            
        Returns:
            List of keywords
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t.lower() not in self.stopwords]
        
        # Count frequencies
        freq = {}
        for token in tokens:
            token_lower = token.lower()
            freq[token_lower] = freq.get(token_lower, 0) + 1
        
        # Sort by frequency
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [token for token, _ in sorted_tokens[:top_k]]
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return self.sentence_splitter.split(text)
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 256,
        overlap: int = 32
    ) -> List[str]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size in words
            overlap: Number of overlapping words
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= chunk_size:
            return [' '.join(tokens)]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(' '.join(chunk_tokens))
            start = end - overlap
        
        return chunks


@dataclass
class ProcessedText:
    """
    Container for processed text with metadata.
    
    Attributes:
        original: Original text
        processed: Processed text
        tokens: List of tokens
        sentences: List of sentences
        keywords: Extracted keywords
        language: Detected language
    """
    original: str
    processed: str
    tokens: List[str]
    sentences: List[str]
    keywords: List[str]
    language: str = 'vi'


def detect_vietnamese(text: str) -> Tuple[bool, float]:
    """
    Detect if text is Vietnamese.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_vietnamese, confidence)
    """
    if not text:
        return False, 0.0
    
    # Count Vietnamese-specific characters
    vn_chars = set(VIETNAMESE_CHARS.lower())
    text_lower = text.lower()
    
    total_alpha = sum(1 for c in text_lower if c.isalpha())
    if total_alpha == 0:
        return False, 0.0
    
    vn_count = sum(1 for c in text_lower if c in vn_chars)
    confidence = vn_count / total_alpha
    
    # Check for common Vietnamese words
    common_words = {'và', 'của', 'là', 'có', 'được', 'trong', 'với', 'này', 'cho', 'một'}
    words = set(text_lower.split())
    word_match = len(words & common_words) / max(len(words), 1)
    
    # Combine scores
    final_confidence = 0.6 * confidence + 0.4 * word_match
    
    return final_confidence > 0.3, final_confidence


def convert_to_ascii_vietnamese(text: str) -> str:
    """
    Convert Vietnamese text to ASCII representation.
    Useful for search indexing.
    
    Args:
        text: Vietnamese text
        
    Returns:
        ASCII representation
    """
    if not text:
        return ""
    
    result = []
    for char in text:
        lower_char = char.lower()
        if lower_char in TONE_MAPPING:
            mapped = TONE_MAPPING[lower_char]
            result.append(mapped if char.islower() else mapped.upper())
        else:
            result.append(char)
    
    return ''.join(result)
