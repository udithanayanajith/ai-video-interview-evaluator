# Video Interview Analyzer

========================

## Author : Uditha Nayanajith - 0766574153

A comprehensive system for evaluating video interview responses using AI analysis, eye tracking, personality assessment, and content evaluation.

## Features

- **Content Analysis:** Evaluates answers against expected keywords
- **AI Detection:** Identifies potential AI-generated responses
- **Eye Tracking:** Measures eye contact and attention metrics
- **Personality Assessment:** Simulates Big Five personality traits
- **Role Suitability:** Scores candidate fit for different technical roles
- **Suspicion Detection:** Flags potential dishonesty or cheating

## Installation

1.  Clone the repository:

        git clone https://github.com/udithanayanajith/ai-video-interview-evaluator.git

2.  Install dependencies:

        pip install -r requirements.txt

## Usage

Start the Flask server:

    python app.py

Send POST requests to `/evaluate` endpoint with:

- Video files (video_one to video_five)
- Question mapping JSON
- Applied role

### Example Request

    curl -X POST -F "video_one=@video1.mp4" -F "video_two=@video2.mp4" \
    -F "mapping={\"question_one\":{\"keywords\":[\"OOP\",\"inheritance\"]}}" \
    -F "applied_role=software_engineer" http://localhost:5000/evaluate

## API Response Structure

The system returns a comprehensive JSON response including:

- Question-by-question content evaluation
- Personality trait analysis
- Eye contact metrics
- AI detection results
- Role suitability scores
- Final composite score

## Configuration

Key configuration options in `app.py`:

- `UPLOAD_FOLDER`: Directory for temporary files
- `MAX_RETRIES`: File operation retry attempts

## Requirements

See `requirements.txt` for complete dependency list.

## Limitations

- Personality traits are currently simulated
- Eye tracking requires clear video of face
- AI detection has some false positive/negative rate

## License

MIT License
