import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import spacy
import textstat
from typing import List, Dict, Union
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ChatbotEvaluator:
    def __init__(self):
        self.rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_score = BERTScorer(lang="en", rescale_with_baseline=True)
        self.nlp = spacy.load("en_core_web_sm")

    def calculate_metrics(self, 
                        original_text: str,
                        generated_text: str,
                        task_type: str = "summarization") -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive metrics for the generated text
        
        Args:
            original_text: Source document
            generated_text: Model generated summary/paraphrase
            task_type: Either "summarization" or "paraphrasing"
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {}

        """
        Calculate comprehensive metrics for the generated text
        
        Args:
            original_text: Source document
            generated_text: Model generated summary/paraphrase
            task_type: Either "summarization" or "paraphrasing"
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {}
        
        # Basic length metrics
        metrics['length_analysis'] = self._analyze_length(original_text, generated_text)
        
        # Content preservation metrics
        metrics['content_preservation'] = self._evaluate_content_preservation(
            original_text, generated_text)
        
        # Readability metrics
        metrics['readability'] = self._analyze_readability(generated_text)
        
        # Semantic similarity
        metrics['semantic_similarity'] = self._calculate_semantic_similarity(
            original_text, generated_text)
        
        # Task-specific metrics
        if task_type == "summarization":
            metrics['compression'] = self._calculate_compression_ratio(
                original_text, generated_text)
        elif task_type == "paraphrasing":
            metrics['lexical_diversity'] = self._calculate_lexical_diversity(
                original_text, generated_text)
            
        return metrics

    def _analyze_length(self, original_text: str, generated_text: str) -> Dict:
        """Analyze length-based metrics"""
        return {
            "original_word_count": len(word_tokenize(original_text)),
            "generated_word_count": len(word_tokenize(generated_text)),
            "original_sentence_count": len(sent_tokenize(original_text)),
            "generated_sentence_count": len(sent_tokenize(generated_text))
        }
    
    def _evaluate_content_preservation(self, original_text: str, generated_text: str) -> Dict:
        """Evaluate how well the content is preserved"""

        # Calculate ROUGE scores
        rouge_scores = self.rouge_score.score(original_text, generated_text)

        # Calculate BERT score
        P, R, F1 = self.bert_score.score([generated_text], [original_text])

         # Calculate BLEU score
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [word_tokenize(original_text)],
            word_tokenize(generated_text),
            smoothing_function=smooth
        )

        return {
            'rouge1_f1': rouge_scores['rouge1'].fmeasure,
            'rouge2_f1': rouge_scores['rouge2'].fmeasure,
            'rougeL_f1': rouge_scores['rougeL'].fmeasure,
            'bert_score_f1': F1.item(),
            'bleu_score': bleu_score
        }
    
    def _analyze_readability(self, text: str) -> Dict:
        """Analyze readability metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog_index': textstat.gunning_fog(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }
    
    def _calculate_semantic_similarity(self, original_text: str, generated_text: str) -> float:
        """Calculate semantic similarity using spaCy"""
        doc1 = self.nlp(original_text)
        doc2 = self.nlp(generated_text)
        return doc1.similarity(doc2)
    
    def _calculate_compression_ratio(self, original_text: str, generated_text: str) -> float:
        """Calculate compression ratio for summarization"""
        original_length = len(word_tokenize(original_text))
        generated_length = len(word_tokenize(generated_text))
        return generated_length / original_length if original_length > 0 else 0
    
    def _calculate_lexical_diversity(self, original_text: str, generated_text: str) -> Dict:
        """Calculate lexical diversity metrics for paraphrasing"""
        original_words = word_tokenize(original_text.lower())
        generated_words = word_tokenize(generated_text.lower())
        
        original_unique = len(set(original_words))
        generated_unique = len(set(generated_words))
        
        return {
            'original_ttr': original_unique / len(original_words) if original_words else 0,
            'generated_ttr': generated_unique / len(generated_words) if generated_words else 0,
            'vocabulary_change': abs(original_unique - generated_unique) / max(original_unique, generated_unique)
        }