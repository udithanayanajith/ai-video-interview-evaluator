import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip

recognizer = sr.Recognizer()

def process_video(file_path, upload_id, output_dir) -> str:
    audio_path = os.path.join(output_dir, f"{upload_id}.wav")
    video = None
    try:
        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile(audio_path, logger=None)
        
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        return text
    except Exception as e:
        print("No Audio found, "f"[ERROR] Processing failed: {str(e)}")
        return ""
    finally:
        # Explicitly close video and audio objects
        if video:
            video.close()
        if 'audio' in locals():
            audio.close()

def evaluate_answer(user_answer: str, keywords: list) -> dict:
    if not user_answer:
        return {
            "score": "0%",
            "keywords_found": [],
            "keywords_missing": keywords,
            "feedback": "No answer detected"
        }

    user_answer = user_answer.lower()
    keywords = [kw.lower() for kw in keywords]
    found = [kw for kw in keywords if kw in user_answer]
    score = len(found) / len(keywords) if keywords else 0

    feedback = (
        "Excellent! You mentioned most key concepts." if score >= 0.8 else
        "Good attempt, but some important concepts were missing." if score >= 0.5 else
        "You missed many key concepts. Please review the material."
    )

    return {
        "score": f"{round(score * 100)}%",
        "keywords_found": found,
        "keywords_missing": [kw for kw in keywords if kw not in found],
        "feedback": feedback
    }
