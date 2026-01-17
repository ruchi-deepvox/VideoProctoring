# Video Proctoring System

AI-powered interview analysis system with support for both pre-recorded videos and **real-time live video sessions**.

## Features

### 🎥 Live Video Analysis (`face_live.py`)
- **Real-time frame analysis** via WebSocket connection
- **Face detection** with multi-face alerts
- **Emotion tracking** (happy, sad, angry, surprised, calm, etc.)
- **Eye contact monitoring** with looking-away detection
- **Posture analysis** based on head position
- **Engagement scoring** in real-time
- **Suspicious object detection** (phones, notes, etc.)
- **Text detection** for potential cheating
- **Live metrics dashboard** with instant feedback
- **Final comprehensive report** at session end

### 📹 Pre-recorded Video Analysis (`face.py`, `face1.py`)
- Download and analyze videos from URLs
- Frame extraction at configurable intervals
- Comprehensive behavioral analysis
- LLM-enhanced insights (OpenAI/Anthropic)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
cp .env.example .env
```

## Environment Variables

Create a `.env` file with:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
OPENAI_API_KEY=your_openai_key  # Optional - for LLM insights
ANTHROPIC_API_KEY=your_anthropic_key  # Optional - alternative LLM
SECRET_KEY=your_flask_secret_key
```

## Running the Servers

### Live Video Server (Port 5001)
```bash
python face_live.py
```

### Pre-recorded Video Server (Port 5000)
```bash
python face.py
# or
python face1.py
```

## Live Video API

### WebSocket Events

Connect to: `ws://localhost:5001/socket.io/`

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Server → Client | Connection established |
| `start_session` | Client → Server | Start new analysis session |
| `session_started` | Server → Client | Session created with ID |
| `analyze_frame` | Client → Server | Send frame for analysis |
| `frame_analysis` | Server → Client | Real-time analysis result |
| `alerts` | Server → Client | Proctoring alerts |
| `get_live_metrics` | Client → Server | Request aggregated metrics |
| `live_metrics` | Server → Client | Current session metrics |
| `end_session` | Client → Server | End session |
| `session_ended` | Server → Client | Final report |

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/start` | Start new session |
| POST | `/api/sessions/{id}/frame` | Analyze single frame |
| GET | `/api/sessions/{id}/metrics` | Get session metrics |
| POST | `/api/sessions/{id}/end` | End session & get report |
| GET | `/api/sessions` | List all sessions |
| GET | `/health` | Health check |

### Example: Start Session (WebSocket)

```javascript
const socket = io('http://localhost:5001');

socket.on('connect', () => {
    socket.emit('start_session', {
        job_data: { jobTitle: 'Software Engineer' }
    });
});

socket.on('session_started', (data) => {
    console.log('Session ID:', data.session_id);
});
```

### Example: Send Frame for Analysis

```javascript
// Capture frame from video element
const canvas = document.createElement('canvas');
canvas.getContext('2d').drawImage(videoElement, 0, 0);
const frameData = canvas.toDataURL('image/jpeg', 0.8);

socket.emit('analyze_frame', {
    session_id: sessionId,
    frame: frameData
});

socket.on('frame_analysis', (data) => {
    console.log('Eye Contact:', data.result.metrics.eye_contact);
    console.log('Engagement:', data.result.metrics.engagement);
    console.log('Dominant Emotion:', data.result.dominant_emotion);
});
```

### Example: REST API (cURL)

```bash
# Start session
curl -X POST http://localhost:5001/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"job_data": {"jobTitle": "Developer"}}'

# Analyze frame (base64)
curl -X POST http://localhost:5001/api/sessions/{session_id}/frame \
  -H "Content-Type: application/json" \
  -d '{"frame": "data:image/jpeg;base64,/9j/4AAQ..."}'

# Get metrics
curl http://localhost:5001/api/sessions/{session_id}/metrics

# End session
curl -X POST http://localhost:5001/api/sessions/{session_id}/end
```

## Demo Client

Open `client_example.html` in a browser to test the live video analysis:

1. Allow camera access when prompted
2. Click "Start Session" to begin
3. Watch real-time metrics update
4. Click "End Session" to see the final report

## Analysis Metrics

### Real-time Metrics (per frame)
- **Eye Contact Score** (0-100): Based on face yaw/pitch angles
- **Engagement Score** (0-100): Composite of expressions + eye contact
- **Facial Expression Score** (0-100): Based on positive emotions
- **Posture Score** (0-100): Based on head stability
- **Confidence Score** (0-100): Face detection confidence

### Alerts
| Alert Type | Severity | Trigger |
|------------|----------|---------|
| `no_face_detected` | Warning | 3+ consecutive frames without face |
| `multiple_faces_detected` | Critical | More than one face in frame |
| `looking_away` | Warning | Yaw > 30° or Pitch > 25° for 5+ frames |
| `low_engagement` | Info | Engagement < 40% for 10+ frames |
| `suspicious_object` | Warning | Phone, paper, etc. detected (>70% confidence) |
| `text_detected` | Info | Significant text visible in frame |

### Final Report Includes
- Session summary (duration, frames analyzed)
- Behavioral analysis scores
- Body language assessment
- Cultural fit analysis
- Emotion timeline
- Complete alert history
- AI-generated summary (if LLM configured)
- Token consumption metrics

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│  Browser/Client │ ◄─────────────────► │  Flask-SocketIO  │
│  (camera feed)  │                     │    Server        │
└─────────────────┘                     └────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  AWS Rekognition │
                                        │  - Face Detection│
                                        │  - Label Detection│
                                        │  - Text Detection │
                                        └────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  LLM (Optional)  │
                                        │  - OpenAI GPT-4  │
                                        │  - Anthropic     │
                                        └──────────────────┘
```

## License

MIT

