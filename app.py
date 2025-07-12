from flask import Flask, request, jsonify
import os
import json
import uuid
import time

from flask_cors import CORS
from video_process.eye_tracking import simulate_eye_tracking_score
from video_process.video_utils import process_video, evaluate_answer
from video_process.personality import simulate_big_five_scores, average_traits, score_roles
from video_process.ai_detection import AIDetector
from video_process.answer_analyzer import AnswerAnalyzer

UPLOAD_FOLDER = 'upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_RETRIES = 3
RETRY_DELAY = 0.5 

app = Flask(__name__)
CORS(app)

ai_detector = AIDetector()  

def cleanup_files(file_list):
    for file_path in file_list:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                break
            except PermissionError:
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f'Failed to delete {file_path} after {MAX_RETRIES} attempts')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
                break

@app.route('/evaluate', methods=['POST'])
def evaluate():
    temp_files = [] 

    try:
        if 'mapping' not in request.form:
            return jsonify({"error": "Missing mapping JSON"}), 400

        try:
            mapping = json.loads(request.form['mapping'])
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in mapping field"}), 400

        applied_role = request.form['applied_role']

        video_keys = ['video_one', 'video_two', 'video_three', 'video_four', 'video_five']
        question_keys = ['question_one', 'question_two', 'question_three', 'question_four', 'question_five']
        print("Role",applied_role)
        results = {}
        personality_traits_list = []
        eye_tracking_scores = []
        transcriptions = {}  
        
        for v_key, q_key in zip(video_keys, question_keys):
            if v_key not in request.files:
                cleanup_files(temp_files)
                return jsonify({"error": f"Missing video: {v_key}"}), 400
            if q_key not in mapping:
                cleanup_files(temp_files)
                return jsonify({"error": f"Missing mapping for: {q_key}"}), 400

            keywords = mapping[q_key].get('keywords', [])
            if not keywords:
                cleanup_files(temp_files)
                return jsonify({"error": f"No keywords provided for {q_key}"}), 400

            # Save video
            file_id = str(uuid.uuid4())
            video_file = request.files[v_key]
            video_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.mp4")
            audio_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.wav")

            video_file.save(video_path)
            temp_files.extend([video_path, audio_path])

            # Step 1: Transcribe & evaluate
            transcription = process_video(video_path, file_id, UPLOAD_FOLDER)
            evaluation_result = evaluate_answer(transcription, keywords)
            evaluation_result["transcription"] = transcription  # Store raw transcription
            results[q_key] = evaluation_result
            transcriptions[q_key] = transcription  # Store for AI detection

            # Step 2: Simulate Big Five traits
            traits = simulate_big_five_scores(video_path)
            personality_traits_list.append(traits)

            # Step 3: Eye tracking
            eye_score = simulate_eye_tracking_score(video_path)
            eye_tracking_scores.append(eye_score)
            

        # Average traits and calculate role suitability
        avg_traits = average_traits(personality_traits_list)
        career_scores = score_roles(avg_traits)

        # Calculate eye tracking metrics
        eye_track_per_question = [{"question": q_key, "score": f"{score:.0f}%" } for q_key, score in zip(question_keys, eye_tracking_scores)]
        avg_eye_tracking_score = round(sum(eye_tracking_scores) / len(eye_tracking_scores), 2)
        ai_results = ai_detector.analyze_responses(transcriptions)
        
        
        
        # Calculate suspicion score
        suspicion_factors = {
            "perfect_answers": all(result["score"] == "100%" for result in results.values()),
            "role_mismatch": (
                applied_role != "software_engineer" and
                max(career_scores.values()) - career_scores.get(applied_role, 0) > 20
            ),
            "low_eye_contact": avg_eye_tracking_score < 30,
            "unnatural_traits": (
                avg_traits.get("Neuroticism", 0) < 0.2 and
                avg_traits.get("Conscientiousness", 0) > 0.9 and
                avg_traits.get("Agreeableness", 0) > 0.9
            )
        }
        suspicion_score = sum(1 for factor in suspicion_factors.values() if factor) * 25
        ai_answer_ratio = sum(1 for result in ai_results.values() if result["detection"] == "AI-generated") / len(ai_results) * 100
        overall_suspicion = min(100, (suspicion_score + ai_answer_ratio) / 2)

        # Final answer calculation
        evaluation_data = {
            "question_results": results,
            "career_scores": career_scores,
            "eye_track_per_question": eye_track_per_question,
            "overall_eye_tracking_score": f"{avg_eye_tracking_score:.2f}%",
            "personality_traits": avg_traits
        }
        
        # ai_detection_results dictionary creation:
        ai_detection_results = {
            "answer_analysis": ai_results,
            "suspicion_factors": suspicion_factors,
            "behavioral_suspicion_score": suspicion_score,  
            "ai_answer_ratio": round(ai_answer_ratio, 1),   
            "overall_suspicion_score": round(overall_suspicion, 1),  
            "is_suspicious": overall_suspicion >= 50,
            "applied_role": applied_role
        }
        
        final_score = AnswerAnalyzer.calculate_final_score(evaluation_data, ai_detection_results)

        # Final result
        return jsonify({
            "success": True,
            "question_results": results,
            "career_scores": career_scores,
            "eye_track_per_question": eye_track_per_question,
            "personality_traits": avg_traits,
            "overall_eye_tracking_score": f"{avg_eye_tracking_score:.2f}%",
            "ai_detection_results": {
                "answer_analysis": ai_results,
                "suspicion_factors": suspicion_factors,
                "behavioral_suspicion_score": f"{suspicion_score:.0f}%",
                "ai_answer_ratio": f"{round(ai_answer_ratio, 1):.0f}%",
                "overall_suspicion_score": f"{round(overall_suspicion, 1):.0f}%",
                "is_suspicious": overall_suspicion >= 50,
                "applied_role": applied_role
            },
            "final_evaluation": final_score
        })

    except Exception as e:
        cleanup_files(temp_files)
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500
    finally:
        cleanup_files(temp_files)

if __name__ == '__main__':
    app.run(debug=True, port=5000)