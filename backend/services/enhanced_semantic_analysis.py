"""
Enhanced Semantic Analysis Module for Aphasia Speech Therapy

This module provides advanced semantic evaluation specifically designed for
aphasic speech patterns, implementing:
- Multi-layer semantic analysis (embeddings, category matching, circumlocution)
- Semantic paraphasia detection using WordNet
- Circumlocution detection for word-finding difficulties
- Clinical classification with aphasia-specific thresholds

Based on clinical research in semantic processing disorders.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Try to import nltk for WordNet
try:
    import nltk
    from nltk.corpus import wordnet as wn
    # Ensure WordNet is downloaded
    try:
        wn.synsets('test')
        WORDNET_AVAILABLE = True
    except LookupError:
        logger.info("Downloading WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        WORDNET_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available - semantic paraphasia detection will be limited")
    WORDNET_AVAILABLE = False
    wn = None

# Load sentence transformer model (bi-encoder for fast similarity)
try:
    # Use more capable model for better semantic matching
    SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
    logger.info("Enhanced semantic model (all-mpnet-base-v2) loaded successfully")
except Exception as e:
    logger.warning(f"Could not load all-mpnet-base-v2, falling back to MiniLM: {e}")
    try:
        SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Fallback semantic model (all-MiniLM-L6-v2) loaded")
    except Exception as e2:
        logger.error(f"Failed to load any semantic model: {e2}")
        SEMANTIC_MODEL = None

# Load cross-encoder for more accurate similarity scoring (used for borderline cases)
CROSS_ENCODER = None
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    logger.info("Cross-encoder model loaded successfully for accurate re-ranking")
except Exception as e:
    logger.warning(f"Cross-encoder not available, using bi-encoder only: {e}")


# Knowledge base for circumlocution detection
OBJECT_FEATURES = {
    'bottle': ['drink', 'water', 'liquid', 'container', 'pour', 'cap', 'plastic', 'glass', 'hold'],
    'cup': ['drink', 'beverage', 'coffee', 'tea', 'handle', 'sip', 'mug', 'ceramic', 'hold'],
    'phone': ['call', 'talk', 'communicate', 'ring', 'dial', 'message', 'text', 'screen', 'mobile'],
    'book': ['read', 'pages', 'story', 'words', 'cover', 'library', 'learn', 'paper', 'write'],
    'chair': ['sit', 'seat', 'furniture', 'legs', 'back', 'rest', 'wood', 'comfortable'],
    'pen': ['write', 'ink', 'paper', 'draw', 'sign', 'cap', 'click', 'hold'],
    'key': ['lock', 'door', 'open', 'metal', 'car', 'house', 'turn', 'unlock'],
    'watch': ['time', 'wrist', 'clock', 'hours', 'minutes', 'wear', 'tell', 'digital'],
    'glasses': ['see', 'eyes', 'read', 'lens', 'wear', 'vision', 'frame', 'look'],
    'spoon': ['eat', 'soup', 'food', 'metal', 'scoop', 'stir', 'mouth', 'kitchen'],
    'fork': ['eat', 'food', 'prongs', 'metal', 'stab', 'dinner', 'mouth', 'kitchen'],
    'knife': ['cut', 'sharp', 'blade', 'food', 'slice', 'metal', 'kitchen', 'handle'],
    'toothbrush': ['teeth', 'brush', 'clean', 'mouth', 'bristles', 'bathroom', 'morning'],
    'umbrella': ['rain', 'wet', 'cover', 'protect', 'open', 'handle', 'weather', 'dry'],
    'comb': ['hair', 'brush', 'teeth', 'style', 'morning', 'bathroom', 'grooming'],
    'shoe': ['foot', 'walk', 'wear', 'lace', 'sole', 'pair', 'step', 'leather'],
    'hat': ['head', 'wear', 'sun', 'cover', 'style', 'brim', 'cap', 'fashion'],
    'bag': ['carry', 'hold', 'things', 'strap', 'pocket', 'zipper', 'shopping', 'travel'],
}

# Semantic category hierarchies for paraphasia detection
SEMANTIC_CATEGORIES = {
    'kitchen_utensil': ['spoon', 'fork', 'knife', 'spatula', 'ladle'],
    'drinking_vessel': ['cup', 'mug', 'glass', 'bottle', 'bowl'],
    'writing_instrument': ['pen', 'pencil', 'marker', 'crayon'],
    'clothing': ['shirt', 'pants', 'shoe', 'hat', 'jacket', 'sock'],
    'furniture': ['chair', 'table', 'bed', 'couch', 'desk'],
    'electronics': ['phone', 'computer', 'television', 'radio', 'tablet'],
    'personal_care': ['toothbrush', 'comb', 'brush', 'razor', 'towel'],
}


@dataclass
class SemanticCategoryMatch:
    """Result of semantic category matching."""
    path_similarity: float
    shared_hypernyms: List[str]
    same_category: bool
    category_name: Optional[str]


@dataclass
class CircumlocutionResult:
    """Result of circumlocution detection."""
    is_circumlocution: bool
    features_preserved: float
    matched_features: List[str]
    clinical_significance: str


@dataclass
class EnhancedSemanticResult:
    """Complete enhanced semantic analysis result."""
    # Core metrics
    classification: str  # 'correct', 'partial', 'circumlocution', 'semantic_paraphasia', 'wrong'
    similarity_score: float
    confidence: float

    # Detailed analysis
    direct_similarity: float
    category_similarity: float
    circumlocution_analysis: Optional[CircumlocutionResult]
    semantic_paraphasia: Optional[Dict]

    # Clinical
    object_mentioned: bool
    explanation: str
    semantic_features: Dict
    clinical_notes: List[str]


def compute_direct_similarity(text1: str, text2: str, use_cross_encoder: bool = True) -> float:
    """
    Compute direct embedding similarity between two texts.

    Uses bi-encoder for fast initial scoring, and cross-encoder for
    more accurate scoring on borderline cases (0.5-0.9 range).

    Args:
        text1: First text
        text2: Second text
        use_cross_encoder: Whether to use cross-encoder for borderline cases

    Returns:
        Cosine similarity score (0-1)
    """
    if not SEMANTIC_MODEL:
        return 0.0

    try:
        # Stage 1: Fast bi-encoder similarity
        embeddings = SEMANTIC_MODEL.encode([text1, text2])
        bi_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        bi_score = max(0.0, min(1.0, float(bi_score)))

        # Stage 2: Cross-encoder for borderline cases (more accurate but slower)
        if use_cross_encoder and CROSS_ENCODER and 0.5 < bi_score < 0.9:
            try:
                cross_score = CROSS_ENCODER.predict([(text1, text2)])[0]
                # Normalize cross-encoder score (typically 0-1 but can vary)
                cross_score = max(0.0, min(1.0, float(cross_score)))
                # Weight: 30% bi-encoder, 70% cross-encoder for borderline cases
                final_score = 0.3 * bi_score + 0.7 * cross_score
                logger.debug(f"Cross-encoder used: bi={bi_score:.3f}, cross={cross_score:.3f}, final={final_score:.3f}")
                return final_score
            except Exception as e:
                logger.debug(f"Cross-encoder failed, using bi-encoder score: {e}")
                return bi_score

        return bi_score

    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


def compute_cross_encoder_score(text1: str, text2: str) -> Optional[float]:
    """
    Compute cross-encoder similarity score directly.

    Cross-encoders are more accurate than bi-encoders but slower
    as they process both texts together.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cross-encoder score (0-1) or None if unavailable
    """
    if not CROSS_ENCODER:
        return None

    try:
        score = CROSS_ENCODER.predict([(text1, text2)])[0]
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.error(f"Cross-encoder scoring failed: {e}")
        return None


def check_semantic_category(expected_word: str, actual_word: str) -> SemanticCategoryMatch:
    """
    Check if words share semantic categories using WordNet.

    This is critical for detecting semantic paraphasias where a patient
    substitutes a semantically related word (e.g., "apple" for "orange").

    Args:
        expected_word: Expected word
        actual_word: Actual word produced

    Returns:
        SemanticCategoryMatch with analysis
    """
    if not WORDNET_AVAILABLE or not wn:
        # Fallback to category dictionary
        for category, words in SEMANTIC_CATEGORIES.items():
            if expected_word.lower() in words and actual_word.lower() in words:
                return SemanticCategoryMatch(
                    path_similarity=0.7,
                    shared_hypernyms=[category],
                    same_category=True,
                    category_name=category
                )
        return SemanticCategoryMatch(
            path_similarity=0.0,
            shared_hypernyms=[],
            same_category=False,
            category_name=None
        )

    try:
        expected_synsets = wn.synsets(expected_word.lower())
        actual_synsets = wn.synsets(actual_word.lower())

        if not expected_synsets or not actual_synsets:
            return SemanticCategoryMatch(
                path_similarity=0.0,
                shared_hypernyms=[],
                same_category=False,
                category_name=None
            )

        # Calculate maximum path similarity
        max_sim = 0.0
        shared_hypernyms = []

        for s1 in expected_synsets[:3]:  # Limit to top 3 senses
            for s2 in actual_synsets[:3]:
                # Path similarity
                sim = s1.path_similarity(s2)
                if sim and sim > max_sim:
                    max_sim = sim

                # Find shared hypernyms
                s1_hypernyms = set(h.name() for h in s1.lowest_common_hypernyms(s2))
                shared_hypernyms.extend(list(s1_hypernyms))

        shared_hypernyms = list(set(shared_hypernyms))

        # Determine if same category
        same_category = max_sim > 0.5 or len(shared_hypernyms) > 0

        # Get category name from most specific shared hypernym
        category_name = None
        if shared_hypernyms:
            # Filter out very general categories
            specific_hypernyms = [h for h in shared_hypernyms
                                  if h not in ['entity.n.01', 'object.n.01', 'whole.n.02']]
            if specific_hypernyms:
                category_name = specific_hypernyms[0].replace('.n.01', '').replace('_', ' ')

        return SemanticCategoryMatch(
            path_similarity=max_sim,
            shared_hypernyms=shared_hypernyms,
            same_category=same_category,
            category_name=category_name
        )

    except Exception as e:
        logger.error(f"Error in semantic category check: {e}")
        return SemanticCategoryMatch(
            path_similarity=0.0,
            shared_hypernyms=[],
            same_category=False,
            category_name=None
        )


def detect_circumlocution(
    expected_word: str,
    actual_phrase: str,
    custom_features: Optional[List[str]] = None
) -> CircumlocutionResult:
    """
    Detect if response is a circumlocution (talking around the word).

    Circumlocution is a word-finding strategy where patients describe
    the target instead of naming it (e.g., "the thing you write with" for "pen").

    Args:
        expected_word: The target word
        actual_phrase: The patient's response
        custom_features: Optional custom feature list for the object

    Returns:
        CircumlocutionResult with analysis
    """
    expected_lower = expected_word.lower()
    actual_lower = actual_phrase.lower()

    # Get features for this object
    if custom_features:
        features = custom_features
    elif expected_lower in OBJECT_FEATURES:
        features = OBJECT_FEATURES[expected_lower]
    else:
        # Generate basic features from WordNet definitions
        features = []
        if WORDNET_AVAILABLE and wn:
            synsets = wn.synsets(expected_lower)
            if synsets:
                # Extract words from definition
                definition = synsets[0].definition()
                features = [w.lower() for w in definition.split()
                           if len(w) > 3 and w.isalpha()][:10]

    if not features:
        return CircumlocutionResult(
            is_circumlocution=False,
            features_preserved=0.0,
            matched_features=[],
            clinical_significance="Unable to analyze - no feature data available"
        )

    # Count feature matches
    matched_features = []
    for feature in features:
        if feature.lower() in actual_lower:
            matched_features.append(feature)

    features_preserved = len(matched_features) / len(features) if features else 0

    # Circumlocution criteria:
    # 1. Multiple features mentioned (>=2)
    # 2. Target word NOT mentioned
    # 3. Response is longer than target word
    target_mentioned = expected_lower in actual_lower
    is_circumlocution = (
        len(matched_features) >= 2 and
        not target_mentioned and
        len(actual_phrase.split()) >= 3
    )

    # Clinical significance
    if is_circumlocution:
        if features_preserved > 0.5:
            clinical = "Strong circumlocution - semantic access preserved, lexical retrieval impaired"
        else:
            clinical = "Partial circumlocution - some semantic access"
    elif target_mentioned:
        clinical = "Target word accessed directly"
    else:
        clinical = "No circumlocution pattern detected"

    return CircumlocutionResult(
        is_circumlocution=is_circumlocution,
        features_preserved=features_preserved,
        matched_features=matched_features,
        clinical_significance=clinical
    )


def detect_semantic_paraphasia(
    expected_word: str,
    actual_words: List[str],
    expected_sentence: Optional[str] = None,
    actual_sentence: Optional[str] = None
) -> Optional[Dict]:
    """
    Detect semantic paraphasias in the response.

    A semantic paraphasia is when a patient substitutes a semantically
    related word (e.g., "orange" for "apple", or "back" for "bag").

    This function uses position-aware matching: it compares the word at the
    same position in the transcript as the target word in the expected sentence.

    Args:
        expected_word: Target word (the object name)
        actual_words: Words in patient's response
        expected_sentence: The expected sentence (e.g., "I use the bag.")
        actual_sentence: The actual transcript (e.g., "I use the back.")

    Returns:
        Dictionary with paraphasia analysis or None
    """
    expected_lower = expected_word.lower()

    # Strategy 1: Position-based matching (preferred)
    # Find the position of expected_word in expected_sentence,
    # then check the word at that position in actual_sentence
    if expected_sentence and actual_sentence:
        expected_words_list = expected_sentence.lower().split()
        actual_words_list = actual_sentence.lower().split()

        # Find position of target word in expected sentence
        target_position = None
        for i, word in enumerate(expected_words_list):
            word_clean = word.strip('.,!?;:"\'')
            if word_clean == expected_lower:
                target_position = i
                break

        if target_position is not None and target_position < len(actual_words_list):
            # Get the word at the same position in actual response
            produced_word = actual_words_list[target_position].strip('.,!?;:"\'')

            # Skip if it's the correct word
            if produced_word == expected_lower:
                return None

            # Check semantic relationship
            category_match = check_semantic_category(expected_word, produced_word)

            # For position-matched words, also check phonetic similarity
            # (e.g., "bag" -> "back" is a phonological paraphasia, not semantic)
            is_phonetically_similar = _check_phonetic_similarity(expected_lower, produced_word)

            if is_phonetically_similar:
                return {
                    'paraphasia_type': 'phonological',
                    'expected_word': expected_word,
                    'produced_word': produced_word,
                    'semantic_similarity': category_match.path_similarity,
                    'shared_category': category_match.category_name,
                    'clinical_significance': f"Phonological paraphasia: '{produced_word}' for '{expected_word}' - words are phonetically similar"
                }
            elif category_match.same_category or category_match.path_similarity > 0.3:
                return {
                    'paraphasia_type': 'semantic',
                    'expected_word': expected_word,
                    'produced_word': produced_word,
                    'semantic_similarity': category_match.path_similarity,
                    'shared_category': category_match.category_name,
                    'clinical_significance': f"Semantic paraphasia: '{produced_word}' for '{expected_word}' - words share category '{category_match.category_name or 'semantic field'}'"
                }
            elif produced_word != expected_lower:
                # Unrelated substitution
                return {
                    'paraphasia_type': 'unrelated',
                    'expected_word': expected_word,
                    'produced_word': produced_word,
                    'semantic_similarity': category_match.path_similarity,
                    'shared_category': None,
                    'clinical_significance': f"Unrelated word substitution: '{produced_word}' for '{expected_word}'"
                }

    # Strategy 2: Fallback - scan all words for semantic relationships
    # This is used when sentence-level position matching isn't available
    for word in actual_words:
        word_lower = word.lower().strip('.,!?;:"\'')

        # Skip if it's the correct word
        if word_lower == expected_lower:
            return None

        # Skip very common words
        if word_lower in ['the', 'a', 'an', 'it', 'is', 'this', 'that', 'i', 'use', 'have', 'get']:
            continue

        # Check semantic relationship
        category_match = check_semantic_category(expected_word, word_lower)

        if category_match.same_category or category_match.path_similarity > 0.3:
            return {
                'paraphasia_type': 'semantic',
                'expected_word': expected_word,
                'produced_word': word_lower,
                'semantic_similarity': category_match.path_similarity,
                'shared_category': category_match.category_name,
                'clinical_significance': f"Semantic paraphasia: '{word_lower}' for '{expected_word}' - words share category '{category_match.category_name or 'semantic field'}'"
            }

    return None


def _check_phonetic_similarity(word1: str, word2: str) -> bool:
    """
    Check if two words are phonetically similar (differ by 1-2 phonemes).

    This helps distinguish phonological paraphasias from semantic ones.
    Examples: bag/back, cat/bat, pen/pin

    Args:
        word1: First word
        word2: Second word

    Returns:
        True if words are phonetically similar
    """
    # Simple heuristic: check edit distance and shared characters
    if len(word1) == 0 or len(word2) == 0:
        return False

    # Same length, differ by 1-2 characters
    if abs(len(word1) - len(word2)) <= 1:
        differences = 0
        for i in range(min(len(word1), len(word2))):
            if word1[i] != word2[i]:
                differences += 1
        # Add length difference
        differences += abs(len(word1) - len(word2))

        # Consider phonetically similar if ≤2 character differences
        if differences <= 2:
            return True

    # Check if words share same beginning and ending
    if len(word1) >= 3 and len(word2) >= 3:
        if word1[:2] == word2[:2] or word1[-2:] == word2[-2:]:
            return True

    return False


def analyze_response_structure(transcript: str, expected: str) -> Dict:
    """
    Analyze the structure of the response for clinical insights.

    Args:
        transcript: Patient's response
        expected: Expected answer

    Returns:
        Dictionary with structural analysis
    """
    transcript_words = transcript.lower().split()
    expected_words = expected.lower().split()

    features = {
        'word_count': len(transcript_words),
        'expected_word_count': len(expected_words),
        'response_length_ratio': len(transcript_words) / max(1, len(expected_words)),
    }

    # Detect uncertainty markers
    uncertainty_markers = ['maybe', 'perhaps', "don't know", 'not sure', 'um', 'uh', 'think']
    features['uncertainty_detected'] = any(m in transcript.lower() for m in uncertainty_markers)

    # Detect false starts
    words = transcript.split()
    false_starts = 0
    for i in range(len(words) - 1):
        if len(words[i]) <= 2 and words[i][0] == words[i+1][0]:
            false_starts += 1
    features['false_starts'] = false_starts

    # Response type classification
    if features['uncertainty_detected']:
        features['response_type'] = 'uncertain'
    elif features['word_count'] > features['expected_word_count'] * 2:
        features['response_type'] = 'elaborate'
    elif features['word_count'] < features['expected_word_count'] * 0.5:
        features['response_type'] = 'truncated'
    else:
        features['response_type'] = 'appropriate_length'

    return features


def evaluate_answer_enhanced(
    expected_answer: str,
    transcript: str,
    object_name: str,
    question_type: Optional[str] = None,
    expected_answers: Optional[List[str]] = None
) -> EnhancedSemanticResult:
    """
    Perform enhanced semantic evaluation with aphasia-specific analysis.

    This is the main entry point for enhanced semantic analysis.

    Args:
        expected_answer: Primary expected answer
        transcript: Patient's transcribed response
        object_name: Object being discussed
        question_type: Type of question (identification, functional, etc.)
        expected_answers: List of acceptable answers (if multiple)

    Returns:
        EnhancedSemanticResult with complete analysis
    """
    clinical_notes = []

    # Handle edge cases
    if not transcript or not transcript.strip():
        return EnhancedSemanticResult(
            classification='wrong',
            similarity_score=0.0,
            confidence=1.0,
            direct_similarity=0.0,
            category_similarity=0.0,
            circumlocution_analysis=None,
            semantic_paraphasia=None,
            object_mentioned=False,
            explanation="No speech detected",
            semantic_features={'response_type': 'silence', 'word_count': 0},
            clinical_notes=["No verbal response - may indicate severe anomia or apraxia"]
        )

    transcript_lower = transcript.lower().strip()
    object_lower = object_name.lower()
    object_mentioned = object_lower in transcript_lower

    # Handle multiple expected answers
    all_expected = [expected_answer]
    if expected_answers:
        all_expected.extend(expected_answers)

    # Layer 1: Direct embedding similarity (best match)
    direct_similarities = []
    for exp in all_expected:
        sim = compute_direct_similarity(exp, transcript)
        direct_similarities.append(sim)
    direct_similarity = max(direct_similarities) if direct_similarities else 0.0

    # Layer 2: Semantic category matching
    transcript_words = transcript_lower.split()
    category_match = check_semantic_category(object_name, transcript_words[0] if transcript_words else "")
    category_similarity = category_match.path_similarity

    # Layer 3: Circumlocution detection
    circum_result = detect_circumlocution(object_name, transcript)

    # Layer 4: Semantic paraphasia detection (position-aware)
    paraphasia = detect_semantic_paraphasia(
        object_name,
        transcript_words,
        expected_sentence=expected_answer,
        actual_sentence=transcript
    )

    # Analyze response structure
    semantic_features = analyze_response_structure(transcript, expected_answer)
    semantic_features['object_mentioned'] = object_mentioned

    # Classification logic (aphasia-specific thresholds)

    # Perfect match
    if direct_similarity > 0.85:
        classification = 'correct'
        confidence = 0.95
        explanation = "Response semantically matches expected answer"
        clinical_notes.append("Accurate semantic response")

    # Circumlocution detected
    elif circum_result.is_circumlocution:
        classification = 'circumlocution'
        confidence = 0.85
        similarity_score = 0.7  # Give partial credit
        explanation = f"Circumlocution detected - patient described {object_name} using features"
        clinical_notes.append(circum_result.clinical_significance)
        clinical_notes.append(f"Features mentioned: {', '.join(circum_result.matched_features)}")

    # Paraphasia detected (semantic, phonological, or unrelated)
    elif paraphasia:
        paraphasia_type = paraphasia.get('paraphasia_type', 'semantic')
        if paraphasia_type == 'phonological':
            classification = 'phonological_paraphasia'
            confidence = 0.85
            similarity_score = 0.7  # Higher score - phonological errors preserve semantics
            explanation = paraphasia['clinical_significance']
            clinical_notes.append(f"Phonological paraphasia: produced '{paraphasia['produced_word']}' for '{object_name}'")
            clinical_notes.append("Words are phonetically similar - lexical access intact, phonological encoding affected")
        elif paraphasia_type == 'unrelated':
            classification = 'wrong'
            confidence = 0.85
            similarity_score = 0.3
            explanation = paraphasia['clinical_significance']
            clinical_notes.append(f"Word substitution: produced '{paraphasia['produced_word']}' for '{object_name}'")
        else:  # semantic
            classification = 'semantic_paraphasia'
            confidence = 0.80
            similarity_score = 0.6
            explanation = paraphasia['clinical_significance']
            clinical_notes.append(f"Semantic paraphasia: produced '{paraphasia['produced_word']}' for '{object_name}'")
            if paraphasia.get('shared_category'):
                clinical_notes.append(f"Words share category: {paraphasia['shared_category']}")

    # Partial match
    elif direct_similarity >= 0.60 or category_similarity > 0.5:
        classification = 'partial'
        confidence = 0.85
        explanation = "Response partially matches expected answer"
        clinical_notes.append("Partial semantic match - may indicate word-finding difficulty")

    # Object identified in identification questions
    elif question_type == 'identification' and object_mentioned:
        classification = 'correct'
        confidence = 1.0
        explanation = f"Correctly identified the {object_name}"
        clinical_notes.append("Target object correctly named")

    # Wrong
    else:
        classification = 'wrong'
        confidence = 0.90
        explanation = "Response does not match expected answer"
        clinical_notes.append("Semantic match not found")

    # Calculate final similarity score
    if classification == 'circumlocution':
        similarity_score = 0.7 * circum_result.features_preserved + 0.3 * direct_similarity
    elif classification == 'semantic_paraphasia':
        similarity_score = 0.6
    elif classification == 'phonological_paraphasia':
        similarity_score = 0.7  # Already set above, but be explicit
    else:
        similarity_score = direct_similarity

    # Add additional clinical insights
    if semantic_features.get('uncertainty_detected'):
        clinical_notes.append("Uncertainty markers detected in response")

    if semantic_features.get('false_starts', 0) > 0:
        clinical_notes.append(f"False starts detected ({semantic_features['false_starts']}) - may indicate word-finding difficulty")

    if semantic_features.get('response_type') == 'elaborate':
        clinical_notes.append("Elaborate response - possible compensation strategy")
    elif semantic_features.get('response_type') == 'truncated':
        clinical_notes.append("Truncated response - possible word-finding or motor planning difficulty")

    return EnhancedSemanticResult(
        classification=classification,
        similarity_score=similarity_score,
        confidence=confidence,
        direct_similarity=direct_similarity,
        category_similarity=category_similarity,
        circumlocution_analysis=circum_result,
        semantic_paraphasia=paraphasia,
        object_mentioned=object_mentioned,
        explanation=explanation,
        semantic_features=semantic_features,
        clinical_notes=clinical_notes
    )


def format_enhanced_semantic_result_for_api(result: EnhancedSemanticResult) -> Dict:
    """
    Format EnhancedSemanticResult for API response.

    Args:
        result: EnhancedSemanticResult object

    Returns:
        Dictionary suitable for JSON serialization
    """
    circum_data = None
    if result.circumlocution_analysis:
        circum_data = {
            'is_circumlocution': result.circumlocution_analysis.is_circumlocution,
            'features_preserved': round(result.circumlocution_analysis.features_preserved, 3),
            'matched_features': result.circumlocution_analysis.matched_features,
            'clinical_significance': result.circumlocution_analysis.clinical_significance
        }

    return {
        'classification': result.classification,
        'similarity_score': round(result.similarity_score, 3),
        'confidence': round(result.confidence, 3),
        'direct_similarity': round(result.direct_similarity, 3),
        'category_similarity': round(result.category_similarity, 3),
        'circumlocution_analysis': circum_data,
        'semantic_paraphasia': result.semantic_paraphasia,
        'object_mentioned': result.object_mentioned,
        'explanation': result.explanation,
        'semantic_features': result.semantic_features,
        'clinical_notes': result.clinical_notes
    }
