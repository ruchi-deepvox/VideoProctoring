# 🎥 Live Video Proctoring API

Real-time AI-powered interview video analysis with face detection, emotion tracking, and behavioral insights.

## ✨ Features

- **Real-time Face Detection** - Detects faces and alerts for no face or multiple faces
- **Emotion Tracking** - Tracks emotions (happy, sad, calm, angry, surprised, etc.)
- **Eye Contact Monitoring** - Measures eye contact and alerts when looking away
- **Posture Analysis** - Analyzes head position and stability
- **Engagement Scoring** - Real-time engagement metrics
- **Suspicious Object Detection** - Detects phones, notes, and other suspicious items
- **Comprehensive Reports** - Final detailed analysis report at session end

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ruchi-deepvox/VideoProctoring.git
cd VideoProctoring
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Create a `.env` file in the project root:

```env
# AWS Credentials (Required for face/emotion detection)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Optional - for AI-generated summaries
OPENAI_API_KEY=your_openai_api_key

# Flask secret key
SECRET_KEY=your_random_secret_key
```

### 4. Run the Server

```bash
python face_live.py
```

Server will start at: **http://localhost:5001**

### 5. Test the API

Open in browser: http://localhost:5001/health

Or test with the demo client: Open `client_example.html` in your browser.

---

## 📡 API Endpoints

### Base URL: `http://localhost:5001`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions/start` | Start a new proctoring session |
| `POST` | `/api/sessions/{id}/frame` | Analyze a video frame |
| `GET` | `/api/sessions/{id}/metrics` | Get current session metrics |
| `GET` | `/api/sessions/{id}/alerts` | Get all session alerts |
| `GET` | `/api/sessions/{id}/emotions` | Get emotion timeline |
| `POST` | `/api/sessions/{id}/end` | End session & get final report |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/docs` | Full API documentation |
| `GET` | `/health` | Health check |

---

## 🔧 Usage Examples

### Start a Session

```bash
curl -X POST http://localhost:5001/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"job_data": {"jobTitle": "Software Engineer"}}'
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc-123-def",
  "status": "active",
  "start_time": "2024-01-17T10:30:00"
}
```

### Analyze a Frame

```bash
curl -X POST http://localhost:5001/api/sessions/{SESSION_ID}/frame \
  -H "Content-Type: application/json" \
  -d '{"frame": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "face_count": 1,
    "metrics": {
      "eye_contact": 85,
      "engagement": 78,
      "confidence": 92,
      "posture": 88
    },
    "dominant_emotion": "CALM",
    "alerts": []
  }
}
```

### Get Session Metrics

```bash
curl http://localhost:5001/api/sessions/{SESSION_ID}/metrics
```

### End Session & Get Report

```bash
curl -X POST http://localhost:5001/api/sessions/{SESSION_ID}/end
```

---

## 🖥️ JavaScript Integration

```javascript
// 1. Start Session
const startResponse = await fetch('http://localhost:5001/api/sessions/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ job_data: { jobTitle: 'Developer' } })
});
const { session_id } = await startResponse.json();

// 2. Capture and Send Frames (every 1 second)
const video = document.getElementById('webcam');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

setInterval(async () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  const frameData = canvas.toDataURL('image/jpeg', 0.8);

  const response = await fetch(`http://localhost:5001/api/sessions/${session_id}/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: frameData })
  });
  
  const result = await response.json();
  console.log('Eye Contact:', result.result.metrics.eye_contact);
  console.log('Emotion:', result.result.dominant_emotion);
}, 1000);

// 3. End Session
const report = await fetch(`http://localhost:5001/api/sessions/${session_id}/end`, {
  method: 'POST'
});
const finalReport = await report.json();
console.log('Final Report:', finalReport);
```

---

## ⚠️ Alert Types

| Alert | Severity | Description |
|-------|----------|-------------|
| `no_face_detected` | Warning | No face visible for 3+ frames |
| `multiple_faces_detected` | Critical | More than one person detected |
| `looking_away` | Warning | Not looking at screen |
| `low_engagement` | Info | Low engagement detected |
| `suspicious_object` | Warning | Phone/notes detected |
| `text_detected` | Info | Text visible in frame |

---

## 📊 Metrics Explained

| Metric | Range | Description |
|--------|-------|-------------|
| `eye_contact` | 0-100 | How well the candidate maintains eye contact |
| `engagement` | 0-100 | Overall engagement level |
| `confidence` | 0-100 | Face detection confidence |
| `posture` | 0-100 | Head position stability |
| `facial_expression` | 0-100 | Positive expression score |

---

## 🧪 Demo Client

A demo HTML client is included for testing:

1. Start the server: `python face_live.py`
2. Open `client_example.html` in your browser
3. Allow camera access
4. Click "Start Session"
5. Watch real-time metrics update
6. Click "End Session" to see the final report

---

## 📁 Project Structure

```
VideoProctoring/
├── face_live.py          # Main API server
├── client_example.html   # Demo client for testing
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

---

## 🔒 Security Notes

- **Never commit `.env` file** to git (it's in `.gitignore`)
- Rotate API keys if accidentally exposed
- Use HTTPS in production
- Add authentication for production use

---

## 📋 Requirements

- Python 3.8+
- AWS Account with Rekognition access
- OpenAI API key (optional, for AI summaries)

---

## 🐛 Troubleshooting

### "AWS credentials not configured"
- Make sure `.env` file exists with valid AWS credentials
- Check there are no spaces before `=` in the `.env` file

### "No frames being analyzed"
- Check AWS credentials are correct
- Verify camera permissions are granted
- Check browser console for errors

### CORS errors
- The API allows all origins by default
- For production, configure specific allowed origins

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
