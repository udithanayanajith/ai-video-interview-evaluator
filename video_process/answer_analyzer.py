from typing import Dict, Any
import statistics

class AnswerAnalyzer:
    @staticmethod
    def calculate_final_score(evaluation_data: Dict[str, Any], ai_detection_results: Dict[str, Any]) -> Dict[str, Any]:

        # Calculate average content score
        question_scores = [
            float(result["score"].strip('%')) 
            for result in evaluation_data["question_results"].values()
        ]
        avg_content_score = statistics.mean(question_scores)
        
        # Calculate behavioral scores
        eye_score = float(evaluation_data["overall_eye_tracking_score"].strip('%'))
        personality_scores = evaluation_data["personality_traits"]
        
        # Calculate AI suspicion impact (0-100, higher is more suspicious)
        ai_suspicion = ai_detection_results["overall_suspicion_score"]
        
        # Calculate final score with AI suspicion penalty
        base_score = (
            0.5 * avg_content_score + 
            0.2 * eye_score + 
            0.3 * (sum(personality_scores.values()) / len(personality_scores) * 100
        )
        )
        # Apply penalty based on AI suspicion (up to 30% reduction)
        final_score = max(0, base_score * (1 - (ai_suspicion / 100 * 0.3)))
        
        return {
            "content_score": round(avg_content_score, 1),
            "eye_contact_score": round(eye_score, 1),
            "personality_score": round((sum(personality_scores.values()) / len(personality_scores)) * 100, 1),
            "ai_suspicion_score": ai_suspicion,
            "base_score": round(base_score, 1),
            "final_score": round(final_score, 1),
        
        }