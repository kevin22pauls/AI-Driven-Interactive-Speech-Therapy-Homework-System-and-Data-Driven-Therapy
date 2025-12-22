"""
Semantic Analysis Module for Speech Therapy System

This module provides semantic evaluation of patient responses using sentence transformers.
It evaluates the correctness of answers by comparing semantic similarity between
expected answers and patient transcripts.

Features:
- Sentence embedding based similarity scoring
- Three-tier classification (correct/partial/wrong)
- Specific linguistic marker tracking for clinical insights
- Robust error handling for production reliability
"""

import logging
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

# Load model once at module level for efficiency
try:
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {e}")
    MODEL = None


def evaluate_answer(
    expected_answer: str,
    transcript: str,
    object_name: str,
    question_type: Optional[str] = None
) -> Dict:
    """
    Evaluate a patient's answer against the expected response using semantic similarity.

    Args:
        expected_answer: The correct/expected answer
        transcript: The patient's transcribed response
        object_name: The name of the object being discussed
        question_type: Optional type of question (e.g., "identification", "functional", "descriptive")

    Returns:
        Dictionary containing:
        - classification: "correct" | "partial" | "wrong"
        - similarity_score: Float between 0 and 1
        - confidence: Float indicating confidence in classification
        - object_mentioned: Boolean indicating if object name appears in transcript
        - explanation: String explaining the classification
        - semantic_features: Dictionary with linguistic markers and insights
    """
    # Handle edge cases
    if not transcript or not transcript.strip():
        return {
            "classification": "wrong",
            "similarity_score": 0.0,
            "confidence": 1.0,
            "object_mentioned": False,
            "explanation": "No speech detected or transcript is empty",
            "semantic_features": {
                "response_type": "silence",
                "word_count": 0
            }
        }

    if not MODEL:
        return {
            "classification": "wrong",
            "similarity_score": 0.0,
            "confidence": 0.0,
            "object_mentioned": False,
            "explanation": "Semantic evaluation model unavailable",
            "semantic_features": {}
        }

    # Normalize inputs
    transcript_lower = transcript.lower().strip()
    object_name_lower = object_name.lower()

    # Check if object is mentioned
    object_mentioned = object_name_lower in transcript_lower

    # Handle special case for variable answers (e.g., color questions)
    if expected_answer.lower() == "variable":
        # For variable answers, check if response is relevant
        color_words = ["red", "blue", "green", "yellow", "white", "black", "brown",
                      "orange", "purple", "pink", "gray", "grey"]
        has_color = any(color in transcript_lower for color in color_words)

        if has_color:
            return {
                "classification": "correct",
                "similarity_score": 1.0,
                "confidence": 0.95,
                "object_mentioned": object_mentioned,
                "explanation": "Valid color response provided",
                "semantic_features": {
                    "response_type": "color_identification",
                    "word_count": len(transcript.split())
                }
            }

    # Detect question type if not provided
    if not question_type:
        if "what is this" in expected_answer.lower() or expected_answer.lower() == object_name_lower:
            question_type = "identification"
        elif "use" in expected_answer.lower() or "for" in expected_answer.lower():
            question_type = "functional"
        else:
            question_type = "descriptive"

    # For object identification questions
    if question_type == "identification":
        if object_mentioned:
            return {
                "classification": "correct",
                "similarity_score": 1.0,
                "confidence": 1.0,
                "object_mentioned": True,
                "explanation": f"Correctly identified the {object_name}",
                "semantic_features": {
                    "response_type": "direct_identification",
                    "word_count": len(transcript.split()),
                    "response_structure": "contains_target_object"
                }
            }
        else:
            # Check for semantic similarity even if exact object not mentioned
            similarity = compute_semantic_similarity(expected_answer, transcript)

            # Check for common substitutions or related terms
            semantic_features = analyze_semantic_substitutions(object_name, transcript)

            if semantic_features["has_substitution"]:
                return {
                    "classification": "wrong",
                    "similarity_score": similarity,
                    "confidence": 0.85,
                    "object_mentioned": False,
                    "explanation": f"Incorrect - said '{semantic_features['substituted_term']}' instead of '{object_name}'",
                    "semantic_features": semantic_features
                }
            else:
                return {
                    "classification": "wrong",
                    "similarity_score": similarity,
                    "confidence": 0.9,
                    "object_mentioned": False,
                    "explanation": f"Object '{object_name}' not identified in response",
                    "semantic_features": {
                        "response_type": "no_identification",
                        "word_count": len(transcript.split())
                    }
                }

    # For functional and descriptive questions, use semantic similarity
    similarity_score = compute_semantic_similarity(expected_answer, transcript)

    # Analyze semantic features for clinical insights
    semantic_features = analyze_semantic_features(
        expected_answer,
        transcript,
        object_name,
        similarity_score
    )

    # Classification based on similarity thresholds
    if similarity_score > 0.85:
        classification = "correct"
        confidence = 0.95
        explanation = "Response semantically matches expected answer"
    elif similarity_score >= 0.60:
        classification = "partial"
        confidence = 0.85
        explanation = "Response partially matches expected answer"
    else:
        classification = "wrong"
        confidence = 0.90
        explanation = "Response does not match expected answer"

    # Adjust classification based on semantic features
    if classification != "correct" and semantic_features.get("has_paraphrase"):
        classification = "partial"
        explanation = "Response uses different words but conveys similar meaning"

    return {
        "classification": classification,
        "similarity_score": float(similarity_score),
        "confidence": float(confidence),
        "object_mentioned": object_mentioned,
        "explanation": explanation,
        "semantic_features": semantic_features
    }


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Generate embeddings
        embeddings = MODEL.encode([text1, text2])

        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))

    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


def analyze_semantic_features(
    expected: str,
    transcript: str,
    object_name: str,
    similarity_score: float
) -> Dict:
    """
    Analyze linguistic and semantic features for clinical insights.

    Args:
        expected: Expected answer
        transcript: Patient's transcript
        object_name: Object being discussed
        similarity_score: Computed similarity score

    Returns:
        Dictionary of semantic features and clinical markers
    """
    features = {
        "word_count": len(transcript.split()),
        "expected_word_count": len(expected.split()),
        "response_length_ratio": len(transcript.split()) / max(1, len(expected.split())),
        "similarity_score": similarity_score
    }

    # Check for paraphrasing
    if similarity_score > 0.70 and similarity_score < 0.85:
        features["has_paraphrase"] = True
        features["paraphrase_quality"] = "moderate" if similarity_score > 0.75 else "weak"
    else:
        features["has_paraphrase"] = False

    # Analyze response structure
    transcript_lower = transcript.lower()
    if any(phrase in transcript_lower for phrase in ["i don't know", "not sure", "maybe"]):
        features["response_type"] = "uncertain"
        features["confidence_level"] = "low"
    elif "?" in transcript:
        features["response_type"] = "questioning"
    else:
        features["response_type"] = "declarative"

    # Check for semantic field accuracy (related terms)
    features["uses_related_terms"] = check_semantic_field(expected, transcript, object_name)

    return features


def analyze_semantic_substitutions(object_name: str, transcript: str) -> Dict:
    """
    Analyze if the patient substituted the object with a related term.

    Args:
        object_name: Expected object name
        transcript: Patient's transcript

    Returns:
        Dictionary with substitution analysis
    """
    # Define common substitutions for speech therapy context
    substitution_map = {
        "cup": ["mug", "glass", "bottle", "container"],
        "bottle": ["cup", "glass", "container", "jar"],
        "phone": ["telephone", "mobile", "cell", "device"],
        # Add more as needed
    }

    transcript_lower = transcript.lower()
    features = {
        "has_substitution": False,
        "substituted_term": None,
        "substitution_type": None
    }

    if object_name.lower() in substitution_map:
        for substitute in substitution_map[object_name.lower()]:
            if substitute in transcript_lower:
                features["has_substitution"] = True
                features["substituted_term"] = substitute
                features["substitution_type"] = "semantic_related"
                break

    return features


def check_semantic_field(expected: str, transcript: str, object_name: str) -> bool:
    """
    Check if the response uses terms from the same semantic field.

    Args:
        expected: Expected answer
        transcript: Patient's transcript
        object_name: Object being discussed

    Returns:
        Boolean indicating if related terms are used
    """
    # Define semantic fields for common objects
    semantic_fields = {
        "cup": ["drink", "beverage", "water", "coffee", "tea", "liquid", "sip"],
        "bottle": ["drink", "water", "liquid", "beverage", "container", "pour"],
        "phone": ["call", "talk", "communicate", "dial", "ring", "message", "contact"],
        # Add more semantic fields as needed
    }

    transcript_lower = transcript.lower()

    if object_name.lower() in semantic_fields:
        field_terms = semantic_fields[object_name.lower()]
        return any(term in transcript_lower for term in field_terms)

    return False