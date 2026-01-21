# 🎥 Live Interview Proctoring API

Real-time AI-powered interview analysis with face verification, emotion tracking, and behavioral insights.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Health Check](#1-health-check)
  - [Start Session](#2-start-session)
  - [Verify Document (ID Verification)](#3-verify-document-id-verification) ⭐ NEW
  - [Get Document Status](#4-get-document-status)
  - [Register Face](#5-register-face)
  - [Analyze Frame](#6-analyze-frame)
  - [Get Live Metrics](#7-get-live-metrics)
  - [Get Alerts](#8-get-alerts)
  - [Get Face Verification Status](#9-get-face-verification-status)
  - [End Session](#10-end-session--get-final-report)
  - [List Sessions](#11-list-all-sessions)
  - [Delete Session](#12-delete-session)
  - [Pre-recorded Video Analysis](#13-pre-recorded-video-analysis)
- [Integration Flow](#integration-flow)
- [JavaScript Integration](#javascript-integration)
- [Alert Types](#alert-types)
- [Error Handling](#error-handling)

---

## Overview

This API provides real-time video analysis for interview proctoring, including:
- Face detection and verification
- Emotion analysis
- Body language assessment
- Behavioral metrics
- Cheating detection (multiple faces, looking away, suspicious objects)

**Base URL:** 
- Local: `http://localhost:5001`
- Production: `https://fexo.deepvox.ai`

---

## Features

| Feature | Description |
|---------|-------------|
| 🪪 **Document Verification** | Verify candidate identity against ID documents (Aadhaar, PAN, Passport, etc.) |
| 🔐 Face Verification | Continuous face matching throughout the interview |
| 😊 Emotion Analysis | Real-time emotion detection (happy, calm, confused, etc.) |
| 👁️ Eye Contact Tracking | Monitor where the candidate is looking |
| 🧍 Posture Analysis | Assess body language and posture |
| ⚠️ Alert System | Real-time alerts for suspicious behavior |
| 📊 Comprehensive Reports | Detailed analysis report at session end |
| 🤖 AI Insights | LLM-powered summary and recommendations |

---

## Quick Start

### Recommended API Flow (with Document Verification):

```
1. POST /api/sessions/start              → Create session
2. POST /api/sessions/{id}/verify-document  → Verify ID document against live face ⭐ NEW
3. POST /api/sessions/{id}/frame         → Send frames (loop every 1 second)
4. GET  /api/sessions/{id}/metrics       → Get live metrics (optional)
5. POST /api/sessions/{id}/end           → End session & get report
```

### Alternative Flow (Face Registration Only):

```
1. POST /api/sessions/start              → Create session
2. POST /api/sessions/{id}/register-face → Register candidate face
3. POST /api/sessions/{id}/frame         → Send frames (loop every 1 second)
4. POST /api/sessions/{id}/end           → End session & get report
```

> **Note:** Document verification automatically registers the face if successful, so you don't need to call register-face separately.

---

## API Reference

### 1. Health Check

Check if the API server is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "Live Interview Analyzer API",
  "version": "2.0.0",
  "timestamp": "2026-01-21T10:00:00.000000",
  "active_sessions": 0,
  "features": {
    "aws_rekognition": true,
    "openai_llm": true,
    "anthropic_llm": false,
    "rest_api_enabled": true,
    "websocket_enabled": true,
    "video_url_analysis": true
  }
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/health
```

---

### 2. Start Session

Create a new interview proctoring session.

**Endpoint:** `POST /api/sessions/start`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "job_data": {
    "jobTitle": "Software Engineer",
    "company": "TechCorp",
    "candidateName": "John Doe",
    "interviewType": "Technical"
  }
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "status": "active",
  "start_time": "2026-01-21T10:00:00.000000",
  "face_verification_required": true,
  "websocket_url": "ws://localhost:5001/socket.io/",
  "api_endpoints": {
    "verify_document": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/verify-document",
    "register_face": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/register-face",
    "analyze_frame": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/frame",
    "get_metrics": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/metrics",
    "get_alerts": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/alerts",
    "get_document_status": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/document-status",
    "get_verification_status": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/verification-status",
    "end_session": "/api/sessions/8ab64dbf-839d-4d3d-91b2-7c99d746119d/end"
  },
  "verification_flow": [
    "1. verify_document - Upload ID document and capture live face",
    "2. register_face - Register face for continuous verification (auto if document verified)",
    "3. analyze_frame - Start interview with frame analysis"
  ]
}
```

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"job_data": {"jobTitle": "Software Engineer"}}'
```

---

### 3. Verify Document (ID Verification) ⭐ NEW

**Verify candidate's identity by comparing face from ID document with live captured face.**

Supported document types:
- Aadhaar Card
- PAN Card
- Passport
- Driving License
- Voter ID
- Other Government ID

**Endpoint:** `POST /api/sessions/{session_id}/verify-document`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "document": "data:image/jpeg;base64,/9j/4AAQ...",
  "live_face": "data:image/jpeg;base64,/9j/4AAQ...",
  "document_type": "aadhaar"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document` | string | Yes | Base64-encoded image of ID document |
| `live_face` | string | Yes | Base64-encoded live face capture |
| `document_type` | string | No | One of: `aadhaar`, `pan`, `passport`, `driving_license`, `voter_id`, `other` |

**Success Response (Verified):**
```json
{
  "success": true,
  "verified": true,
  "similarity": 94.5,
  "threshold": 80.0,
  "document_type": "aadhaar",
  "document_face_confidence": 99.2,
  "live_face_confidence": 99.8,
  "message": "Identity verified successfully! Document face matches live face.",
  "face_registered": true,
  "next_step": "You can now proceed with the interview."
}
```

**Success Response (Not Verified):**
```json
{
  "success": true,
  "verified": false,
  "similarity": 45.2,
  "threshold": 80.0,
  "document_type": "aadhaar",
  "document_face_confidence": 98.5,
  "live_face_confidence": 99.1,
  "message": "Identity verification failed. Face similarity (45.2%) is below threshold (80%).",
  "next_step": "Please try again with a clearer document or better lighting."
}
```

**Error Responses:**

No face in document:
```json
{
  "success": false,
  "verified": false,
  "error": "No face detected in the document. Please upload a clear image of your ID with a visible photo.",
  "document_type": "aadhaar"
}
```

Multiple faces in live image:
```json
{
  "success": false,
  "verified": false,
  "error": "Multiple faces detected in live image. Please ensure only you are in the frame.",
  "document_type": "aadhaar"
}
```

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/api/sessions/{session_id}/verify-document \
  -H "Content-Type: application/json" \
  -d '{
    "document": "data:image/jpeg;base64,/9j/4AAQ...",
    "live_face": "data:image/jpeg;base64,/9j/4AAQ...",
    "document_type": "aadhaar"
  }'
```

> **Note:** If document verification succeeds, the live face is automatically registered for continuous verification during the interview. You don't need to call `/register-face` separately.

---

### 4. Get Document Status

Get the current document verification status for a session.

**Endpoint:** `GET /api/sessions/{session_id}/document-status`

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "document_verified": true,
  "document_type": "aadhaar",
  "verification_result": {
    "verified": true,
    "similarity": 94.5,
    "document_type": "aadhaar",
    "timestamp": "2026-01-21T10:00:30.000000"
  },
  "face_registered": true
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/api/sessions/{session_id}/document-status
```

---

### 5. Register Face (Alternative to Document Verification)

**⚠️ REQUIRED** - Register candidate's face before starting frame analysis. This enables identity verification throughout the interview.

**Endpoint:** `POST /api/sessions/{session_id}/register-face`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "face": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
}
```

> **Note:** The `face` field should contain a base64-encoded image. You can include or exclude the `data:image/jpeg;base64,` prefix - both formats are accepted.

**Success Response:**
```json
{
  "success": true,
  "message": "Face registered successfully",
  "face_confidence": 99.5,
  "face_bounding_box": {
    "Width": 0.35,
    "Height": 0.45,
    "Left": 0.32,
    "Top": 0.15
  },
  "verification_enabled": true
}
```

**Error Responses:**

No face detected:
```json
{
  "success": false,
  "error": "No face detected in the image. Please ensure your face is clearly visible."
}
```

Multiple faces:
```json
{
  "success": false,
  "error": "Multiple faces detected. Please ensure only you are in the frame."
}
```

Low quality:
```json
{
  "success": false,
  "error": "Face detection confidence too low (75.5%). Please ensure good lighting and face the camera directly."
}
```

Already registered:
```json
{
  "success": false,
  "error": "Face already registered for this session",
  "face_registered": true
}
```

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/api/sessions/{session_id}/register-face \
  -H "Content-Type: application/json" \
  -d '{"face": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

---

### 6. Analyze Frame

Send a video frame for real-time analysis. Call this endpoint every 1-2 seconds during the interview.

**Endpoint:** `POST /api/sessions/{session_id}/frame`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "result": {
    "frame_number": 45,
    "timestamp": 45.2,
    "face_detected": true,
    "face_count": 1,
    "emotions": {
      "HAPPY": 15.5,
      "CALM": 65.2,
      "CONFUSED": 8.3,
      "SURPRISED": 5.1,
      "SAD": 3.2,
      "ANGRY": 1.5,
      "DISGUSTED": 0.8,
      "FEAR": 0.4
    },
    "dominant_emotion": "CALM",
    "metrics": {
      "eye_contact": 85,
      "confidence": 78,
      "engagement": 82,
      "posture": 90
    },
    "face_verification": {
      "verified": true,
      "similarity": 97.5,
      "message": "Face verified",
      "threshold": 80.0
    },
    "pose": {
      "pitch": -5.2,
      "roll": 2.1,
      "yaw": 8.3
    },
    "labels_detected": ["Person", "Suit", "Tie", "Laptop"],
    "alerts": []
  }
}
```

**Response with Alerts:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "result": {
    "frame_number": 46,
    "timestamp": 46.2,
    "face_detected": true,
    "face_count": 2,
    "face_verification": {
      "verified": false,
      "similarity": 45.2,
      "message": "Different person detected",
      "threshold": 80.0
    },
    "alerts": [
      {
        "type": "multiple_faces",
        "severity": "critical",
        "message": "Multiple faces detected (2 faces)",
        "timestamp": 46.2,
        "face_count": 2
      },
      {
        "type": "face_mismatch",
        "severity": "critical",
        "message": "Face verification failed 3 consecutive times",
        "timestamp": 46.2
      }
    ]
  }
}
```

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/api/sessions/{session_id}/frame \
  -H "Content-Type: application/json" \
  -d '{"frame": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

---

### 7. Get Live Metrics

Get aggregated metrics for the current session.

**Endpoint:** `GET /api/sessions/{session_id}/metrics`

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "metrics": {
    "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
    "frames_analyzed": 120,
    "total_frames_received": 125,
    "session_duration": 125.5,
    "is_active": true,
    "metrics": {
      "avg_eye_contact": 82,
      "avg_confidence": 78,
      "avg_engagement": 85,
      "avg_posture": 88,
      "avg_facial_expressions": 75,
      "dominant_emotion": "CALM",
      "emotion_distribution": {
        "CALM": 45,
        "HAPPY": 30,
        "CONFUSED": 15,
        "OTHER": 10
      }
    },
    "alert_count": 3
  }
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/api/sessions/{session_id}/metrics
```

---

### 8. Get Alerts

Get all alerts generated during the session.

**Endpoint:** `GET /api/sessions/{session_id}/alerts`

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "total_alerts": 5,
  "alerts": [
    {
      "type": "looking_away",
      "severity": "warning",
      "message": "Candidate looking away from camera",
      "timestamp": 12.5
    },
    {
      "type": "no_face_detected",
      "severity": "warning",
      "message": "No face detected in frame",
      "timestamp": 25.3
    },
    {
      "type": "multiple_faces",
      "severity": "critical",
      "message": "Multiple faces detected (2 faces)",
      "timestamp": 45.8,
      "face_count": 2
    },
    {
      "type": "face_mismatch",
      "severity": "critical",
      "message": "Face verification failed 3 consecutive times",
      "timestamp": 67.2
    },
    {
      "type": "suspicious_object",
      "severity": "warning",
      "message": "Suspicious object detected: Cell Phone",
      "timestamp": 89.1,
      "confidence": 95.2
    }
  ]
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/api/sessions/{session_id}/alerts
```

---

### 9. Get Face Verification Status

Get detailed face verification statistics.

**Endpoint:** `GET /api/sessions/{session_id}/verification-status`

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "face_registered": true,
  "verification_enabled": true,
  "verification_summary": {
    "enabled": true,
    "registered": true,
    "total_verifications": 120,
    "successful_verifications": 115,
    "average_similarity": 94.5,
    "max_similarity": 99.2,
    "min_similarity": 65.3,
    "verification_rate": 95.8,
    "total_mismatches": 5
  },
  "recent_verifications": [
    {
      "timestamp": 118.2,
      "verified": true,
      "similarity": 96.5,
      "message": "Face verified",
      "threshold": 80.0
    },
    {
      "timestamp": 119.2,
      "verified": true,
      "similarity": 97.8,
      "message": "Face verified",
      "threshold": 80.0
    },
    {
      "timestamp": 120.2,
      "verified": true,
      "similarity": 95.2,
      "message": "Face verified",
      "threshold": 80.0
    }
  ]
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/api/sessions/{session_id}/verification-status
```

---

### 10. End Session & Get Final Report

End the interview session and receive a comprehensive analysis report.

**Endpoint:** `POST /api/sessions/{session_id}/end`

**Response:**
```json
{
  "success": true,
  "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
  "report": {
    "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
    "session_summary": {
      "start_time": "2026-01-21T10:00:00.000000",
      "end_time": "2026-01-21T10:05:00.000000",
      "duration_seconds": 300.5,
      "total_frames_received": 300,
      "frames_analyzed": 295,
      "total_alerts": 5
    },
    "behavioral_analysis": {
      "eye_contact": 85,
      "posture": 88,
      "gestures": 75,
      "facial_expressions": 82,
      "voice_tone": 78,
      "confidence": 80,
      "engagement": 85,
      "emotion_summary": {
        "CALM": 45.5,
        "HAPPY": 32.3,
        "CONFUSED": 12.1,
        "SURPRISED": 5.2,
        "SAD": 2.8,
        "ANGRY": 1.2,
        "DISGUSTED": 0.5,
        "FEAR": 0.4
      }
    },
    "body_language_analysis": {
      "summary": "Good body language with appropriate posture and gestures",
      "overallAveragePercentage": 82,
      "detailed_scores": {
        "posture": 88,
        "gestures": 75,
        "eye_contact": 85,
        "facial_expressions": 82,
        "head_stability": 80
      }
    },
    "cultural_fit_analysis": {
      "summary": "Good cultural fit with appropriate interview attire",
      "overallAveragePercentage": 85
    },
    "overall_behavior_analysis": {
      "overallAveragePercentage": 83,
      "summary": "Good interview performance with professional demeanor"
    },
    "video_analysis_insights": {
      "positive_indicators": [
        "Maintains strong eye contact throughout",
        "Professional posture and demeanor",
        "Appropriate facial expressions"
      ],
      "areas_for_improvement": [
        "Could show more enthusiasm and energy",
        "Occasional looking away from camera"
      ],
      "recommendations": [
        "Practice maintaining consistent eye contact",
        "Prepare specific examples to demonstrate expertise",
        "Research company culture to align presentation"
      ]
    },
    "face_verification_summary": {
      "enabled": true,
      "registered": true,
      "total_verifications": 295,
      "successful_verifications": 290,
      "average_similarity": 94.5,
      "max_similarity": 99.2,
      "min_similarity": 65.3,
      "verification_rate": 98.3,
      "total_mismatches": 5
    },
    "emotion_timeline": [
      {"timestamp": 1.0, "dominant_emotion": "CALM", "emotions": {"CALM": 70, "HAPPY": 20}},
      {"timestamp": 2.0, "dominant_emotion": "CALM", "emotions": {"CALM": 65, "HAPPY": 25}}
    ],
    "alert_history": [
      {
        "type": "looking_away",
        "severity": "warning",
        "message": "Candidate looking away from camera",
        "timestamp": 45.2
      }
    ],
    "ai_summary": {
      "summary": "The candidate demonstrated strong professional presence throughout the interview with consistent eye contact and confident demeanor. Body language was generally positive with appropriate gestures.",
      "keyStrengths": [
        "Excellent eye contact maintained throughout",
        "Confident and professional demeanor",
        "Appropriate attire for interview setting"
      ],
      "areasOfGrowth": [
        "Could show more enthusiasm when discussing achievements",
        "Occasional nervous gestures noticed"
      ],
      "overallVideoScore": 85
    },
    "token_consumption": {
      "total_api_calls": 2,
      "total_tokens_consumed": 850,
      "total_prompt_tokens": 650,
      "total_completion_tokens": 200,
      "average_tokens_per_call": 425
    },
    "analysis_metadata": {
      "llm_enhanced": true,
      "llm_provider": "openai",
      "face_verification_enabled": true,
      "face_registered": true
    }
  }
}
```

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/api/sessions/{session_id}/end
```

---

### 11. List All Sessions

Get a list of all active and ended sessions.

**Endpoint:** `GET /api/sessions`

**Response:**
```json
{
  "success": true,
  "total_sessions": 3,
  "sessions": [
    {
      "session_id": "8ab64dbf-839d-4d3d-91b2-7c99d746119d",
      "is_active": true,
      "start_time": "2026-01-21T10:00:00.000000",
      "frames_analyzed": 120,
      "alert_count": 2
    },
    {
      "session_id": "abc12345-6789-def0-1234-567890abcdef",
      "is_active": false,
      "start_time": "2026-01-21T09:30:00.000000",
      "frames_analyzed": 300,
      "alert_count": 5
    }
  ]
}
```

**cURL:**
```bash
curl https://fexo.deepvox.ai/api/sessions
```

---

### 12. Delete Session

Delete a session and its data.

**Endpoint:** `DELETE /api/sessions/{session_id}`

**Response:**
```json
{
  "success": true,
  "message": "Session deleted successfully"
}
```

**cURL:**
```bash
curl -X DELETE https://fexo.deepvox.ai/api/sessions/{session_id}
```

---

### 13. Pre-recorded Video Analysis

Analyze a pre-recorded video file (for legacy UI compatibility).

**Endpoint:** `POST /comprehensive-interview-analysis`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "video_url": "https://example.com/interview-video.mp4",
  "questionsWithAnswer": [
    {
      "question": "Tell me about yourself",
      "answer": "I am a software engineer with 5 years of experience..."
    }
  ],
  "jobData": {
    "jobTitle": "Senior Software Engineer",
    "company": "TechCorp"
  },
  "frame_interval": 60,
  "llm_provider": "openai"
}
```

**Response:** Same structure as End Session report.

**cURL:**
```bash
curl -X POST https://fexo.deepvox.ai/comprehensive-interview-analysis \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4"}'
```

---

## Integration Flow

### Complete Integration Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR UI APPLICATION                       │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Start Session                                            │
│ POST /api/sessions/start                                         │
│ → Receive session_id                                             │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Register Candidate Face                                  │
│ POST /api/sessions/{session_id}/register-face                    │
│ → Capture photo, send as base64                                  │
│ → Wait for success: true                                         │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Interview Loop (every 1-2 seconds)                       │
│ POST /api/sessions/{session_id}/frame                            │
│ → Send video frame as base64                                     │
│ → Receive metrics, emotions, face verification, alerts           │
│ → Update UI with real-time data                                  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                         (Optional polling)
                                   │
┌─────────────────────────────────────────────────────────────────┐
│ Step 3b: Get Aggregated Metrics (optional)                       │
│ GET /api/sessions/{session_id}/metrics                           │
│ GET /api/sessions/{session_id}/alerts                            │
│ GET /api/sessions/{session_id}/verification-status               │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: End Session                                              │
│ POST /api/sessions/{session_id}/end                              │
│ → Receive comprehensive final report                             │
│ → Display results to user                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## JavaScript Integration

### Complete Example

```javascript
class InterviewProctoring {
  constructor(apiBaseUrl) {
    this.API_BASE = apiBaseUrl || 'https://fexo.deepvox.ai';
    this.sessionId = null;
    this.captureInterval = null;
  }

  // 1. Start a new session
  async startSession(jobData = {}) {
    const response = await fetch(`${this.API_BASE}/api/sessions/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_data: jobData })
    });
    
    const data = await response.json();
    if (data.success) {
      this.sessionId = data.session_id;
    }
    return data;
  }

  // 2. Register candidate's face
  async registerFace(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    const base64Image = canvas.toDataURL('image/jpeg', 0.9);

    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/register-face`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face: base64Image })
      }
    );
    
    return await response.json();
  }

  // 3. Capture and send a frame
  async captureFrame(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    const base64Image = canvas.toDataURL('image/jpeg', 0.8);

    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/frame`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: base64Image })
      }
    );
    
    return await response.json();
  }

  // 4. Start continuous frame capture
  startCapturing(videoElement, intervalMs = 1000, onFrameResult) {
    this.captureInterval = setInterval(async () => {
      if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        const result = await this.captureFrame(videoElement);
        if (onFrameResult) {
          onFrameResult(result);
        }
      }
    }, intervalMs);
  }

  // 5. Stop capturing
  stopCapturing() {
    if (this.captureInterval) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
  }

  // 6. Get current metrics
  async getMetrics() {
    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/metrics`
    );
    return await response.json();
  }

  // 7. Get alerts
  async getAlerts() {
    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/alerts`
    );
    return await response.json();
  }

  // 8. Get face verification status
  async getVerificationStatus() {
    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/verification-status`
    );
    return await response.json();
  }

  // 9. End session and get report
  async endSession() {
    this.stopCapturing();
    
    const response = await fetch(
      `${this.API_BASE}/api/sessions/${this.sessionId}/end`,
      { method: 'POST' }
    );
    
    const data = await response.json();
    this.sessionId = null;
    return data;
  }
}

// Usage Example
const proctoring = new InterviewProctoring('https://fexo.deepvox.ai');

async function runInterview() {
  // Get video element
  const video = document.getElementById('videoElement');
  
  // Start session
  const session = await proctoring.startSession({
    jobTitle: 'Software Engineer',
    candidateName: 'John Doe'
  });
  console.log('Session started:', session.session_id);
  
  // Register face
  const faceResult = await proctoring.registerFace(video);
  if (!faceResult.success) {
    alert('Face registration failed: ' + faceResult.error);
    return;
  }
  console.log('Face registered:', faceResult.face_confidence + '%');
  
  // Start capturing frames
  proctoring.startCapturing(video, 1000, (result) => {
    // Update UI with each frame's analysis
    console.log('Frame result:', result);
    
    // Check for alerts
    if (result.result?.alerts?.length > 0) {
      console.warn('Alerts:', result.result.alerts);
    }
    
    // Check face verification
    if (result.result?.face_verification?.verified === false) {
      console.error('Face mismatch detected!');
    }
  });
  
  // End session after interview (e.g., on button click)
  // const report = await proctoring.endSession();
}
```

### React Integration Example

```jsx
import { useState, useRef, useEffect } from 'react';

function InterviewProctoring() {
  const [sessionId, setSessionId] = useState(null);
  const [faceRegistered, setFaceRegistered] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);
  const intervalRef = useRef(null);
  
  const API_BASE = 'https://fexo.deepvox.ai';

  const startSession = async () => {
    const res = await fetch(`${API_BASE}/api/sessions/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_data: { jobTitle: 'Developer' } })
    });
    const data = await res.json();
    setSessionId(data.session_id);
    return data.session_id;
  };

  const registerFace = async (sid) => {
    const canvas = document.createElement('canvas');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const res = await fetch(`${API_BASE}/api/sessions/${sid}/register-face`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ face: canvas.toDataURL('image/jpeg', 0.9) })
    });
    const data = await res.json();
    setFaceRegistered(data.success);
    return data;
  };

  const captureFrame = async () => {
    if (!sessionId || !videoRef.current) return;
    
    const canvas = document.createElement('canvas');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: canvas.toDataURL('image/jpeg', 0.8) })
    });
    const data = await res.json();
    setMetrics(data.result);
  };

  const startInterview = async () => {
    const sid = await startSession();
    await registerFace(sid);
    setIsRecording(true);
    intervalRef.current = setInterval(captureFrame, 1000);
  };

  const endInterview = async () => {
    clearInterval(intervalRef.current);
    const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/end`, {
      method: 'POST'
    });
    const data = await res.json();
    setIsRecording(false);
    console.log('Final Report:', data.report);
  };

  return (
    <div>
      <video ref={videoRef} autoPlay muted />
      <button onClick={startInterview} disabled={isRecording}>
        Start Interview
      </button>
      <button onClick={endInterview} disabled={!isRecording}>
        End Interview
      </button>
      {metrics && (
        <div>
          <p>Eye Contact: {metrics.metrics?.eye_contact}%</p>
          <p>Confidence: {metrics.metrics?.confidence}%</p>
          <p>Verified: {metrics.face_verification?.verified ? '✓' : '✗'}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Alert Types

| Type | Severity | Description | Trigger |
|------|----------|-------------|---------|
| `no_face_detected` | warning | No face visible | 3+ consecutive frames |
| `multiple_faces` | critical | More than one face | Immediate |
| `looking_away` | warning | Candidate not looking at camera | 5+ consecutive frames |
| `face_mismatch` | critical | Face doesn't match registered identity | 3+ consecutive mismatches |
| `low_engagement` | info | Low engagement score | 10+ consecutive frames |
| `suspicious_object` | warning | Phone, notes, etc. detected | Confidence > 70% |
| `text_detected` | info | Readable text in frame | Confidence > 80% |

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Session not found |
| 500 | Server error |

### Error Response Format

```json
{
  "success": false,
  "error": "Detailed error message"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid session" | Session ID doesn't exist | Start a new session |
| "Session has ended" | Trying to analyze on ended session | Start a new session |
| "No frame data provided" | Missing frame in request | Include base64 image |
| "Face already registered" | Duplicate registration attempt | Use existing registration |
| "AWS Rekognition not initialized" | Server config issue | Check server logs |

---

## Rate Limits & Best Practices

1. **Frame Interval**: Send frames every 1-2 seconds (not faster)
2. **Image Quality**: JPEG at 80% quality is optimal
3. **Image Size**: 720p resolution is sufficient
4. **Session Cleanup**: Always call end session when done
5. **Error Handling**: Implement retry logic for network errors

---

## Support

For issues or questions, contact the development team.

**Version:** 2.0.0  
**Last Updated:** January 2026
