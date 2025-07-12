from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the AI detection model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "openai-community/roberta-base-openai-detector"
            ).to(self.device)
            self.model.eval()
            logger.info("AI detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AI detection model: {e}")
            raise

    def detect_text(self, text: str) -> Tuple[str, float]:
        """Detect if text is AI-generated with modified empty text handling"""
        if not text:  # Empty string case
            return "Empty", 0.0
        if len(text) < 10:  # Short text case
            return "TooShort", 0.0
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            probs = torch.softmax(logits, dim=1).squeeze()
            ai_prob = probs[1].item()
            
            label = "AI-generated" if ai_prob > 0.5 else "Human-written"
            return label, ai_prob
        except Exception as e:
            logger.error(f"AI detection error: {e}")
            return "Error", 0.0

    def analyze_responses(self, responses: Dict[str, str]) -> Dict[str, Dict]:
        """Analyze multiple responses with proper empty text handling"""
        results = {}
        for response_id, text in responses.items():
            label, prob = self.detect_text(text)
            human_prob = 0.0 if label in ["Empty", "TooShort", "Error"] else (1 - prob)
            
            results[response_id] = {
                "detection": label,
                "ai_probability": f"{round(prob * 100, 1)}" ,
                "human_probability":f"{round(human_prob * 100, 1)}"

            }
            logger.info(f"Analyzed response {response_id}: {results[response_id]}")
        return results