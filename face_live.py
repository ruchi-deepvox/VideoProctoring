from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import boto3
import json
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import time
import logging
from typing import Dict, List, Any, Optional
import threading
from collections import defaultdict
import uuid
import requests
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# Initialize SocketIO with CORS allowed for all origins
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                    ping_timeout=60, ping_interval=25)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Store active sessions
active_sessions: Dict[str, 'LiveInterviewSession'] = {}
session_lock = threading.Lock()


# CORS Configuration - Belt and suspenders approach
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With,Accept,Origin,X-API-Key,Cache-Control,Pragma'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS,PATCH,HEAD'
    response.headers['Access-Control-Allow-Credentials'] = 'false'
    response.headers['Access-Control-Max-Age'] = '86400'
    response.headers['Vary'] = 'Origin'
    return response


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get('Origin', '*')
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With,Accept,Origin,X-API-Key,Cache-Control,Pragma'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS,PATCH,HEAD'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response


class TokenTracker:
    """Class to track LLM token consumption"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.call_details = []

    def add_call(self, call_type: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.api_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.call_details.append({
            'call_number': self.api_calls,
            'call_type': call_type,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        })

    def get_summary(self):
        return {
            'total_api_calls': self.api_calls,
            'total_tokens_consumed': self.total_tokens,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'average_tokens_per_call': round(self.total_tokens / self.api_calls) if self.api_calls > 0 else 0
        }


class LiveInterviewSession:
    """Manages a live interview session with real-time analysis"""
    
    def __init__(self, session_id: str, job_data: Dict = None):
        self.session_id = session_id
        self.job_data = job_data or {}
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True
        
        # Analysis results storage
        self.frame_analyses: List[Dict] = []
        self.emotion_history: List[Dict] = []
        self.pose_history: List[Dict] = []
        self.alert_history: List[Dict] = []
        
        # Aggregated metrics
        self.emotion_counts: Dict[str, List[float]] = defaultdict(list)
        self.confidence_scores: List[float] = []
        self.eye_contact_scores: List[float] = []
        self.pose_variations: List[Dict] = []
        self.facial_expression_scores: List[float] = []
        self.engagement_scores: List[float] = []
        
        # Clothing/label detection
        self.detected_labels: List[str] = []
        self.formal_scores: List[float] = []
        self.casual_scores: List[float] = []
        
        # Frame counter
        self.frame_count = 0
        self.analyzed_frame_count = 0
        
        # Alert thresholds
        self.alert_thresholds = {
            'no_face_detected': 3,  # Alert after 3 consecutive frames with no face
            'multiple_faces': 1,     # Alert immediately if multiple faces
            'looking_away': 5,       # Alert if looking away for 5 frames
            'low_engagement': 10,    # Alert if low engagement for 10 frames
        }
        
        # Consecutive counters for alerts
        self.no_face_counter = 0
        self.looking_away_counter = 0
        self.low_engagement_counter = 0
        
        # Initialize AWS client
        try:
            self.rekognition = boto3.client(
                'rekognition',
                region_name='us-east-1',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"Session {session_id}: AWS Rekognition initialized")
        except Exception as e:
            logger.error(f"Session {session_id}: Failed to initialize AWS: {e}")
            self.rekognition = None
        
        # Initialize LLM client
        self.llm_provider = None
        self.token_tracker = TokenTracker()
        
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                self.llm_provider = 'openai'
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        elif ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic
                self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
                self.llm_provider = 'anthropic'
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
    
    def analyze_frame(self, frame_data: bytes) -> Dict:
        """Analyze a single frame and return real-time results"""
        self.frame_count += 1
        timestamp = (datetime.now() - self.start_time).total_seconds()
        
        if not self.rekognition:
            return {'error': 'AWS Rekognition not initialized'}
        
        try:
            # Detect faces
            face_response = self.rekognition.detect_faces(
                Image={'Bytes': frame_data},
                Attributes=['ALL']
            )
            
            # Detect labels (clothing, objects)
            label_response = self.rekognition.detect_labels(
                Image={'Bytes': frame_data},
                MaxLabels=30,
                MinConfidence=60
            )
            
            # Detect text (for cheating detection - notes, phones, etc.)
            try:
                text_response = self.rekognition.detect_text(
                    Image={'Bytes': frame_data}
                )
            except:
                text_response = {'TextDetections': []}
            
            # Process results
            analysis_result = self._process_frame_analysis(
                face_response, label_response, text_response, timestamp
            )
            
            self.analyzed_frame_count += 1
            self.frame_analyses.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': timestamp,
                'frame_number': self.frame_count
            }
    
    def _process_frame_analysis(self, face_response: Dict, label_response: Dict, 
                                 text_response: Dict, timestamp: float) -> Dict:
        """Process raw AWS responses into structured analysis"""
        
        faces = face_response.get('FaceDetails', [])
        labels = label_response.get('Labels', [])
        texts = text_response.get('TextDetections', [])
        
        result = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'face_count': len(faces),
            'alerts': [],
            'metrics': {},
            'emotions': {},
            'labels_detected': []
        }
        
        # Check for no face or multiple faces
        if len(faces) == 0:
            self.no_face_counter += 1
            if self.no_face_counter >= self.alert_thresholds['no_face_detected']:
                alert = {
                    'type': 'no_face_detected',
                    'severity': 'warning',
                    'message': 'No face detected in frame',
                    'timestamp': timestamp
                }
                result['alerts'].append(alert)
                self.alert_history.append(alert)
            return result
        else:
            self.no_face_counter = 0
        
        if len(faces) > 1:
            alert = {
                'type': 'multiple_faces_detected',
                'severity': 'critical',
                'message': f'{len(faces)} faces detected - possible unauthorized person',
                'timestamp': timestamp
            }
            result['alerts'].append(alert)
            self.alert_history.append(alert)
        
        # Process primary face
        face = faces[0]
        
        # Extract emotions
        emotions = face.get('Emotions', [])
        emotion_data = {}
        dominant_emotion = None
        max_confidence = 0
        
        for emotion in emotions:
            emotion_type = emotion.get('Type', '')
            confidence = emotion.get('Confidence', 0)
            emotion_data[emotion_type] = confidence
            self.emotion_counts[emotion_type].append(confidence)
            
            if confidence > max_confidence:
                max_confidence = confidence
                dominant_emotion = emotion_type
        
        result['emotions'] = emotion_data
        result['dominant_emotion'] = dominant_emotion
        
        self.emotion_history.append({
            'timestamp': timestamp,
            'emotions': emotion_data,
            'dominant': dominant_emotion
        })
        
        # Extract pose (for eye contact and posture)
        pose = face.get('Pose', {})
        yaw = pose.get('Yaw', 0)
        pitch = pose.get('Pitch', 0)
        roll = pose.get('Roll', 0)
        
        self.pose_variations.append({
            'yaw': yaw, 'pitch': pitch, 'roll': roll, 'timestamp': timestamp
        })
        self.pose_history.append({
            'timestamp': timestamp,
            'yaw': yaw, 'pitch': pitch, 'roll': roll
        })
        
        # Calculate eye contact score
        eye_contact_score = max(0, min(100, 100 - abs(yaw) * 0.8 - abs(pitch) * 0.6))
        self.eye_contact_scores.append(eye_contact_score)
        result['metrics']['eye_contact'] = round(eye_contact_score)
        
        # Check for looking away
        if abs(yaw) > 30 or abs(pitch) > 25:
            self.looking_away_counter += 1
            if self.looking_away_counter >= self.alert_thresholds['looking_away']:
                alert = {
                    'type': 'looking_away',
                    'severity': 'warning',
                    'message': 'Candidate appears to be looking away from screen',
                    'timestamp': timestamp,
                    'yaw': yaw,
                    'pitch': pitch
                }
                result['alerts'].append(alert)
                self.alert_history.append(alert)
        else:
            self.looking_away_counter = 0
        
        # Calculate facial expression score
        happy_conf = emotion_data.get('HAPPY', 0)
        calm_conf = emotion_data.get('CALM', 0)
        facial_expr_score = min(100, (happy_conf + calm_conf) * 1.2)
        self.facial_expression_scores.append(facial_expr_score)
        result['metrics']['facial_expression'] = round(facial_expr_score)
        
        # Calculate engagement score
        engagement_score = min(100, facial_expr_score * 0.6 + eye_contact_score * 0.4)
        self.engagement_scores.append(engagement_score)
        result['metrics']['engagement'] = round(engagement_score)
        
        # Check for low engagement
        if engagement_score < 40:
            self.low_engagement_counter += 1
            if self.low_engagement_counter >= self.alert_thresholds['low_engagement']:
                alert = {
                    'type': 'low_engagement',
                    'severity': 'info',
                    'message': 'Low engagement detected',
                    'timestamp': timestamp
                }
                result['alerts'].append(alert)
                self.alert_history.append(alert)
        else:
            self.low_engagement_counter = 0
        
        # Store confidence score
        face_confidence = face.get('Confidence', 0)
        self.confidence_scores.append(face_confidence)
        result['metrics']['confidence'] = round(face_confidence)
        
        # Calculate posture score
        posture_score = max(60, min(100, 95 - (abs(pitch) * 0.8 + abs(roll) * 0.6)))
        result['metrics']['posture'] = round(posture_score)
        
        # Process labels for clothing/environment
        formal_keywords = ['suit', 'blazer', 'shirt', 'tie', 'dress shirt', 'collar', 
                          'business', 'formal', 'professional', 'jacket', 'dress', 'blouse']
        casual_keywords = ['t-shirt', 'tshirt', 'hoodie', 'sweatshirt', 'tank top', 
                          'polo', 'casual', 'jeans', 'shorts']
        suspicious_keywords = ['phone', 'mobile', 'paper', 'book', 'notes', 'screen', 'monitor']
        
        frame_formal_score = 0
        frame_casual_score = 0
        
        for label in labels:
            label_name = label.get('Name', '').lower()
            confidence = label.get('Confidence', 0)
            result['labels_detected'].append(label['Name'])
            self.detected_labels.append(label['Name'])
            
            if any(keyword in label_name for keyword in formal_keywords):
                frame_formal_score += confidence
            elif any(keyword in label_name for keyword in casual_keywords):
                frame_casual_score += confidence
            
            # Check for suspicious items
            if any(keyword in label_name for keyword in suspicious_keywords):
                if confidence > 70:
                    alert = {
                        'type': 'suspicious_object',
                        'severity': 'warning',
                        'message': f'Suspicious object detected: {label["Name"]}',
                        'timestamp': timestamp,
                        'confidence': confidence
                    }
                    result['alerts'].append(alert)
                    self.alert_history.append(alert)
        
        self.formal_scores.append(frame_formal_score)
        self.casual_scores.append(frame_casual_score)
        
        # Check for detected text (potential cheating)
        if texts:
            significant_texts = [t for t in texts if t.get('Confidence', 0) > 80 
                               and len(t.get('DetectedText', '')) > 5]
            if significant_texts:
                alert = {
                    'type': 'text_detected',
                    'severity': 'info',
                    'message': f'Text detected in frame: {len(significant_texts)} items',
                    'timestamp': timestamp
                }
                result['alerts'].append(alert)
                self.alert_history.append(alert)
        
        return result
    
    def get_live_metrics(self) -> Dict:
        """Get current aggregated metrics for live display"""
        if not self.frame_analyses:
            return {
                'frames_analyzed': 0,
                'session_duration': 0,
                'metrics': {}
            }
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'session_id': self.session_id,
            'frames_analyzed': self.analyzed_frame_count,
            'total_frames_received': self.frame_count,
            'session_duration': round(duration, 1),
            'is_active': self.is_active,
            'metrics': {
                'avg_eye_contact': round(sum(self.eye_contact_scores) / len(self.eye_contact_scores)) if self.eye_contact_scores else 0,
                'avg_engagement': round(sum(self.engagement_scores) / len(self.engagement_scores)) if self.engagement_scores else 0,
                'avg_facial_expression': round(sum(self.facial_expression_scores) / len(self.facial_expression_scores)) if self.facial_expression_scores else 0,
                'avg_confidence': round(sum(self.confidence_scores) / len(self.confidence_scores)) if self.confidence_scores else 0,
            },
            'emotion_summary': self._get_emotion_summary(),
            'alert_count': len(self.alert_history),
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }
    
    def _get_emotion_summary(self) -> Dict:
        """Get summary of detected emotions"""
        summary = {}
        for emotion, values in self.emotion_counts.items():
            if values:
                summary[emotion] = round(sum(values) / len(values), 1)
        return summary
    
    def end_session(self) -> Dict:
        """End the session and generate final report"""
        self.is_active = False
        self.end_time = datetime.now()
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive final report"""
        duration = (self.end_time or datetime.now()) - self.start_time
        
        # Calculate behavioral metrics
        behavioral_analysis = self._calculate_behavioral_metrics()
        body_language_analysis = self._analyze_body_language()
        cultural_fit_analysis = self._analyze_cultural_fit()
        overall_behavior_analysis = self._calculate_overall_behavior(behavioral_analysis)
        video_insights = self._generate_video_insights(behavioral_analysis, body_language_analysis)
        
        # Generate LLM-enhanced summary if available
        ai_summary = self._generate_ai_summary(behavioral_analysis, body_language_analysis)
        
        return {
            'session_id': self.session_id,
            'session_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': (self.end_time or datetime.now()).isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_frames_received': self.frame_count,
                'frames_analyzed': self.analyzed_frame_count,
                'total_alerts': len(self.alert_history)
            },
            'behavioral_analysis': behavioral_analysis,
            'body_language_analysis': {
                'summary': self._get_body_language_summary(body_language_analysis.get('overall_score', 70)),
                'overallAveragePercentage': body_language_analysis.get('overall_score', 70),
                'detailed_scores': body_language_analysis.get('scores', {})
            },
            'cultural_fit_analysis': cultural_fit_analysis,
            'overall_behavior_analysis': overall_behavior_analysis,
            'video_analysis_insights': video_insights,
            'emotion_timeline': self.emotion_history[-50:],  # Last 50 emotion readings
            'alert_history': self.alert_history,
            'ai_summary': ai_summary,
            'token_consumption': self.token_tracker.get_summary(),
            'analysis_metadata': {
                'llm_enhanced': self.llm_provider is not None,
                'llm_provider': self.llm_provider
            }
        }
    
    def _calculate_behavioral_metrics(self) -> Dict:
        """Calculate behavioral metrics from accumulated data"""
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 70
        avg_eye_contact = sum(self.eye_contact_scores) / len(self.eye_contact_scores) if self.eye_contact_scores else 70
        avg_facial_expressions = sum(self.facial_expression_scores) / len(self.facial_expression_scores) if self.facial_expression_scores else 70
        avg_engagement = sum(self.engagement_scores) / len(self.engagement_scores) if self.engagement_scores else 70
        
        posture_score = self._calculate_posture_score()
        gestures_score = self._calculate_gestures_score()
        voice_tone_score = min(100, (avg_confidence + avg_engagement) / 2)
        
        return {
            "eye_contact": round(avg_eye_contact),
            "posture": round(posture_score),
            "gestures": round(gestures_score),
            "facial_expressions": round(avg_facial_expressions),
            "voice_tone": round(voice_tone_score),
            "confidence": round(avg_confidence),
            "engagement": round(avg_engagement),
            "emotion_summary": self._get_emotion_summary()
        }
    
    def _calculate_posture_score(self) -> float:
        if not self.pose_variations:
            return 85
        
        pitches = [abs(p['pitch']) for p in self.pose_variations]
        rolls = [abs(p['roll']) for p in self.pose_variations]
        
        avg_pitch = sum(pitches) / len(pitches)
        avg_roll = sum(rolls) / len(rolls)
        
        return max(60, min(100, 95 - (avg_pitch * 0.8 + avg_roll * 0.6)))
    
    def _calculate_gestures_score(self) -> float:
        if not self.pose_variations:
            return 80
        
        yaws = [p['yaw'] for p in self.pose_variations]
        significant_movements = len([y for y in yaws if abs(y) > 8])
        
        total_frames = len(self.pose_variations)
        movement_ratio = significant_movements / total_frames if total_frames > 0 else 0
        
        if 0.15 <= movement_ratio <= 0.35:
            return 90
        elif 0.1 <= movement_ratio < 0.15:
            return 85
        elif 0.35 < movement_ratio <= 0.5:
            return 80
        else:
            return 75
    
    def _analyze_body_language(self) -> Dict:
        if not self.pose_variations:
            return {
                'scores': {'posture': 0, 'gestures': 0, 'eye_contact': 0, 
                          'facial_expressions': 0, 'head_stability': 0},
                'overall_score': 0
            }
        
        head_positions = self.pose_variations
        gesture_patterns = [p for p in self.pose_variations if abs(p['yaw']) > 5 or abs(p['pitch']) > 5]
        
        body_language_scores = {
            'posture': self._calculate_posture_score(),
            'gestures': self._calculate_gestures_score(),
            'eye_contact': sum(self.eye_contact_scores) / len(self.eye_contact_scores) if self.eye_contact_scores else 70,
            'facial_expressions': sum(self.facial_expression_scores) / len(self.facial_expression_scores) if self.facial_expression_scores else 70,
            'head_stability': self._calculate_head_stability(head_positions)
        }
        
        weights = {'posture': 0.25, 'eye_contact': 0.25, 'gestures': 0.20,
                   'facial_expressions': 0.15, 'head_stability': 0.15}
        
        overall_score = sum(score * weights.get(aspect, 0) for aspect, score in body_language_scores.items())
        
        return {
            'scores': {k: round(v) for k, v in body_language_scores.items()},
            'overall_score': round(overall_score)
        }
    
    def _calculate_head_stability(self, head_positions: List[Dict]) -> float:
        if not head_positions:
            return 80
        
        yaws = [pos['yaw'] for pos in head_positions]
        pitches = [pos['pitch'] for pos in head_positions]
        
        yaw_variance = sum((y - sum(yaws)/len(yaws))**2 for y in yaws) / len(yaws) if yaws else 0
        pitch_variance = sum((p - sum(pitches)/len(pitches))**2 for p in pitches) / len(pitches) if pitches else 0
        
        total_variance = yaw_variance + pitch_variance
        
        if total_variance < 25:
            return 95
        elif total_variance < 100:
            return 85
        elif total_variance < 225:
            return 75
        else:
            return max(60, 75 - (total_variance - 225) * 0.1)
    
    def _analyze_cultural_fit(self) -> Dict:
        avg_formal = sum(self.formal_scores) / len(self.formal_scores) if self.formal_scores else 0
        avg_casual = sum(self.casual_scores) / len(self.casual_scores) if self.casual_scores else 0
        
        if avg_formal > avg_casual and avg_formal >= 100:
            return {
                "summary": "Excellent cultural fit with appropriate professional dress code",
                "overallAveragePercentage": 90
            }
        elif avg_formal > 50:
            return {
                "summary": "Good cultural fit with appropriate interview attire",
                "overallAveragePercentage": 75
            }
        else:
            return {
                "summary": "Moderate cultural fit - consider more formal attire for interviews",
                "overallAveragePercentage": 65
            }
    
    def _calculate_overall_behavior(self, behavioral_scores: Dict) -> Dict:
        weights = {
            'eye_contact': 0.20, 'confidence': 0.20, 'engagement': 0.15,
            'facial_expressions': 0.15, 'posture': 0.12, 'voice_tone': 0.10, 'gestures': 0.08
        }
        
        weighted_score = sum(score * weights.get(aspect, 0) 
                           for aspect, score in behavioral_scores.items() 
                           if aspect != 'emotion_summary')
        overall_percentage = round(weighted_score)
        
        if overall_percentage >= 85:
            summary = "Strong interview performance with excellent behavioral presence"
        elif overall_percentage >= 75:
            summary = "Good interview performance with professional demeanor"
        else:
            summary = "Interview performance needs improvement in behavioral presentation"
        
        return {
            "overallAveragePercentage": overall_percentage,
            "summary": summary
        }
    
    def _get_body_language_summary(self, overall_score: float) -> str:
        if overall_score >= 85:
            return "Strong non-verbal communication indicating confidence and engagement"
        elif overall_score >= 75:
            return "Good body language with appropriate posture and gestures"
        elif overall_score >= 65:
            return "Moderate body language performance with some areas for improvement"
        else:
            return "Body language needs improvement in multiple areas"
    
    def _generate_video_insights(self, behavioral_analysis: Dict, body_language_analysis: Dict) -> Dict:
        positive_indicators = []
        areas_for_improvement = []
        recommendations = []
        
        for aspect, score in behavioral_analysis.items():
            if aspect == 'emotion_summary':
                continue
            if score >= 85:
                if aspect == 'confidence':
                    positive_indicators.append("Maintains strong composure and self-assurance throughout")
                elif aspect == 'eye_contact':
                    positive_indicators.append("Consistently maintains appropriate eye contact")
                elif aspect == 'posture':
                    positive_indicators.append("Maintains professional posture throughout")
            elif score < 70:
                if aspect == 'eye_contact':
                    areas_for_improvement.append("Work on maintaining more consistent eye contact")
                elif aspect == 'posture':
                    areas_for_improvement.append("Focus on maintaining more upright posture")
                elif aspect == 'engagement':
                    areas_for_improvement.append("Show more enthusiasm and engagement")
        
        # Add alert-based insights
        alert_types = [a['type'] for a in self.alert_history]
        if alert_types.count('looking_away') > 3:
            areas_for_improvement.append("Minimize looking away from camera during interview")
        if alert_types.count('no_face_detected') > 2:
            areas_for_improvement.append("Ensure proper camera positioning for consistent visibility")
        
        if not recommendations:
            recommendations = [
                "Practice mock interviews with video recording",
                "Research the company culture to align presentation",
                "Prepare specific examples to demonstrate expertise"
            ]
        
        return {
            "positive_indicators": positive_indicators[:3],
            "areas_for_improvement": areas_for_improvement[:3],
            "recommendations": recommendations[:3]
        }
    
    def _generate_ai_summary(self, behavioral_analysis: Dict, body_language_analysis: Dict) -> Optional[Dict]:
        if not self.llm_provider:
            return None
        
        prompt = f"""
        Based on the following live interview video analysis data, provide a comprehensive evaluation summary:
        
        Session Duration: {(self.end_time or datetime.now() - self.start_time).total_seconds():.0f} seconds
        Frames Analyzed: {self.analyzed_frame_count}
        Total Alerts: {len(self.alert_history)}
        
        Behavioral Metrics:
        - Eye Contact: {behavioral_analysis.get('eye_contact', 0)}%
        - Confidence: {behavioral_analysis.get('confidence', 0)}%
        - Engagement: {behavioral_analysis.get('engagement', 0)}%
        - Posture: {behavioral_analysis.get('posture', 0)}%
        - Facial Expressions: {behavioral_analysis.get('facial_expressions', 0)}%
        
        Body Language Score: {body_language_analysis.get('overall_score', 0)}%
        
        Alert Summary: {dict((a['type'], alert_types.count(a['type'])) for a in self.alert_history) if self.alert_history else 'No alerts'}
        
        Provide your response in JSON format:
        {{
            "summary": "Brief overview of candidate's video presence",
            "keyStrengths": ["strength 1", "strength 2"],
            "areasOfGrowth": ["improvement area 1", "improvement area 2"],
            "overallVideoScore": 75
        }}
        """
        
        alert_types = [a['type'] for a in self.alert_history]
        
        try:
            if self.llm_provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert interview coach analyzing video interview performance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                usage = response.usage
                self.token_tracker.add_call('ai_summary', usage.prompt_tokens, 
                                           usage.completion_tokens, usage.total_tokens)
                
                result = response.choices[0].message.content
                
            elif self.llm_provider == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                usage = response.usage
                self.token_tracker.add_call('ai_summary', usage.input_tokens,
                                           usage.output_tokens, usage.input_tokens + usage.output_tokens)
                
                result = response.content[0].text
            
            # Parse JSON response
            cleaned = result.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1]
            
            return json.loads(cleaned)
            
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return None


# ==================== SOCKET.IO EVENT HANDLERS ====================

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    
    # Clean up any sessions associated with this client
    with session_lock:
        sessions_to_end = [sid for sid, session in active_sessions.items() 
                         if hasattr(session, 'client_sid') and session.client_sid == request.sid]
        for session_id in sessions_to_end:
            if active_sessions[session_id].is_active:
                active_sessions[session_id].end_session()
            logger.info(f"Cleaned up session {session_id} for disconnected client")


@socketio.on('start_session')
def handle_start_session(data):
    """Start a new live interview session"""
    session_id = data.get('session_id') or str(uuid.uuid4())
    job_data = data.get('job_data', {})
    
    with session_lock:
        if session_id in active_sessions and active_sessions[session_id].is_active:
            emit('session_error', {'error': 'Session already active', 'session_id': session_id})
            return
        
        session = LiveInterviewSession(session_id, job_data)
        session.client_sid = request.sid
        active_sessions[session_id] = session
    
    join_room(session_id)
    
    logger.info(f"Started session: {session_id}")
    emit('session_started', {
        'session_id': session_id,
        'status': 'active',
        'start_time': session.start_time.isoformat()
    })


@socketio.on('analyze_frame')
def handle_analyze_frame(data):
    """Analyze a single frame from live video"""
    session_id = data.get('session_id')
    frame_data = data.get('frame')  # Base64 encoded image
    
    if not session_id or session_id not in active_sessions:
        emit('frame_error', {'error': 'Invalid or inactive session'})
        return
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        emit('frame_error', {'error': 'Session has ended'})
        return
    
    try:
        # Decode base64 frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        
        # Analyze frame
        result = session.analyze_frame(frame_bytes)
        
        # Emit real-time analysis result
        emit('frame_analysis', {
            'session_id': session_id,
            'result': result
        })
        
        # If there are alerts, emit them separately for immediate attention
        if result.get('alerts'):
            emit('alerts', {
                'session_id': session_id,
                'alerts': result['alerts']
            }, room=session_id)
        
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        emit('frame_error', {'error': str(e), 'session_id': session_id})


@socketio.on('get_live_metrics')
def handle_get_live_metrics(data):
    """Get current live metrics for a session"""
    session_id = data.get('session_id')
    
    if not session_id or session_id not in active_sessions:
        emit('metrics_error', {'error': 'Invalid session'})
        return
    
    metrics = active_sessions[session_id].get_live_metrics()
    emit('live_metrics', metrics)


@socketio.on('end_session')
def handle_end_session(data):
    """End a live interview session and get final report"""
    session_id = data.get('session_id')
    questions_with_answers = data.get('questionsWithAnswer', [])
    
    if not session_id or session_id not in active_sessions:
        emit('session_error', {'error': 'Invalid session'})
        return
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        emit('session_error', {'error': 'Session already ended'})
        return
    
    # Generate final report
    report = session.end_session()
    
    leave_room(session_id)
    
    logger.info(f"Ended session: {session_id}")
    emit('session_ended', {
        'session_id': session_id,
        'report': report
    })


# ==================== REST API ENDPOINTS ====================

@app.route('/api/sessions/start', methods=['POST'])
def api_start_session():
    """REST API to start a new session"""
    data = request.get_json() or {}
    session_id = data.get('session_id') or str(uuid.uuid4())
    job_data = data.get('job_data', {})
    
    with session_lock:
        if session_id in active_sessions and active_sessions[session_id].is_active:
            return jsonify({
                'success': False, 
                'error': 'Session already active', 
                'session_id': session_id
            }), 400
        
        session = LiveInterviewSession(session_id, job_data)
        active_sessions[session_id] = session
    
    logger.info(f"Started session via REST: {session_id}")
    return jsonify({
        'success': True,
        'session_id': session_id,
        'status': 'active',
        'start_time': session.start_time.isoformat(),
        'websocket_url': f'ws://{request.host}/socket.io/',
        'api_endpoints': {
            'analyze_frame': f'/api/sessions/{session_id}/frame',
            'get_metrics': f'/api/sessions/{session_id}/metrics',
            'get_alerts': f'/api/sessions/{session_id}/alerts',
            'end_session': f'/api/sessions/{session_id}/end'
        }
    })


@app.route('/api/sessions/<session_id>/frame', methods=['POST'])
def api_analyze_frame(session_id):
    """REST API to analyze a single frame"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        return jsonify({'success': False, 'error': 'Session has ended'}), 400
    
    # Get frame from request
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'frame' not in request.files:
                return jsonify({'success': False, 'error': 'No frame file provided'}), 400
            frame_bytes = request.files['frame'].read()
        else:
            data = request.get_json()
            if not data or 'frame' not in data:
                return jsonify({'success': False, 'error': 'No frame data provided'}), 400
            
            frame_data = data['frame']
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            frame_bytes = base64.b64decode(frame_data)
        
        result = session.analyze_frame(frame_bytes)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'result': result
        })
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sessions/<session_id>/metrics', methods=['GET'])
def api_get_metrics(session_id):
    """REST API to get current session metrics"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    metrics = active_sessions[session_id].get_live_metrics()
    return jsonify({
        'success': True,
        **metrics
    })


@app.route('/api/sessions/<session_id>/end', methods=['POST'])
def api_end_session(session_id):
    """REST API to end a session and get final report"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        return jsonify({'success': False, 'error': 'Session already ended'}), 400
    
    report = session.end_session()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'report': report
    })


@app.route('/api/sessions', methods=['GET'])
def api_list_sessions():
    """List all active sessions"""
    sessions_info = []
    for session_id, session in active_sessions.items():
        sessions_info.append({
            'session_id': session_id,
            'is_active': session.is_active,
            'start_time': session.start_time.isoformat(),
            'frames_analyzed': session.analyzed_frame_count,
            'alert_count': len(session.alert_history)
        })
    
    return jsonify({
        'success': True,
        'total_sessions': len(active_sessions),
        'sessions': sessions_info
    })


@app.route('/api/sessions/<session_id>/alerts', methods=['GET'])
def api_get_alerts(session_id):
    """Get all alerts for a session"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    return jsonify({
        'success': True,
        'session_id': session_id,
        'total_alerts': len(session.alert_history),
        'alerts': session.alert_history
    })


@app.route('/api/sessions/<session_id>/emotions', methods=['GET'])
def api_get_emotions(session_id):
    """Get emotion history for a session"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    return jsonify({
        'success': True,
        'session_id': session_id,
        'emotion_summary': session._get_emotion_summary(),
        'emotion_timeline': session.emotion_history[-100:]  # Last 100 readings
    })


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """Delete a session"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    with session_lock:
        if active_sessions[session_id].is_active:
            active_sessions[session_id].end_session()
        del active_sessions[session_id]
    
    return jsonify({
        'success': True,
        'message': f'Session {session_id} deleted'
    })


@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API Documentation endpoint"""
    return jsonify({
        'api_name': 'Live Video Proctoring API',
        'version': '1.0.0',
        'base_url': request.host_url.rstrip('/'),
        'description': 'Real-time AI-powered interview video analysis with face detection, emotion tracking, and behavioral insights',
        
        'authentication': 'No authentication required (add your own auth middleware for production)',
        
        'endpoints': {
            'Session Management': {
                'POST /api/sessions/start': {
                    'description': 'Start a new proctoring session',
                    'request_body': {
                        'session_id': '(optional) Custom session ID',
                        'job_data': '(optional) Job information object'
                    },
                    'response': {
                        'session_id': 'Unique session identifier',
                        'status': 'active',
                        'start_time': 'ISO timestamp'
                    }
                },
                'POST /api/sessions/{session_id}/end': {
                    'description': 'End session and get final comprehensive report',
                    'response': 'Full analysis report with behavioral metrics'
                },
                'DELETE /api/sessions/{session_id}': {
                    'description': 'Delete a session from memory'
                },
                'GET /api/sessions': {
                    'description': 'List all sessions'
                }
            },
            'Frame Analysis': {
                'POST /api/sessions/{session_id}/frame': {
                    'description': 'Analyze a single video frame',
                    'content_type': 'application/json OR multipart/form-data',
                    'request_body': {
                        'frame': 'Base64 encoded image (data:image/jpeg;base64,... or raw base64)'
                    },
                    'response': {
                        'face_count': 'Number of faces detected',
                        'metrics': {
                            'eye_contact': '0-100 score',
                            'engagement': '0-100 score',
                            'confidence': '0-100 score',
                            'posture': '0-100 score'
                        },
                        'emotions': 'Emotion confidence scores',
                        'dominant_emotion': 'Primary detected emotion',
                        'alerts': 'Any triggered alerts'
                    }
                }
            },
            'Metrics & Monitoring': {
                'GET /api/sessions/{session_id}/metrics': {
                    'description': 'Get current aggregated session metrics'
                },
                'GET /api/sessions/{session_id}/alerts': {
                    'description': 'Get all alerts for a session'
                },
                'GET /api/sessions/{session_id}/emotions': {
                    'description': 'Get emotion history timeline'
                }
            },
            'System': {
                'GET /health': 'Health check with feature status',
                'GET /api/docs': 'This documentation'
            }
        },
        
        'websocket': {
            'url': f'ws://{request.host}/socket.io/',
            'events': {
                'emit': {
                    'start_session': 'Start new session',
                    'analyze_frame': 'Send frame for analysis',
                    'get_live_metrics': 'Request current metrics',
                    'end_session': 'End session'
                },
                'listen': {
                    'connected': 'Connection established',
                    'session_started': 'Session created',
                    'frame_analysis': 'Frame analysis result',
                    'alerts': 'Real-time alerts',
                    'live_metrics': 'Aggregated metrics',
                    'session_ended': 'Final report'
                }
            }
        },
        
        'alert_types': {
            'no_face_detected': 'No face visible in frame (warning)',
            'multiple_faces_detected': 'More than one person detected (critical)',
            'looking_away': 'Candidate not looking at screen (warning)',
            'low_engagement': 'Low engagement detected (info)',
            'suspicious_object': 'Phone/notes detected (warning)',
            'text_detected': 'Text visible in frame (info)'
        },
        
        'integration_example': {
            'javascript': '''
// 1. Start Session
const response = await fetch('http://localhost:5001/api/sessions/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_data: { jobTitle: 'Developer' } })
});
const { session_id } = await response.json();

// 2. Send Frames (from webcam)
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.drawImage(videoElement, 0, 0);
const frameData = canvas.toDataURL('image/jpeg', 0.8);

const analysis = await fetch(`http://localhost:5001/api/sessions/${session_id}/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: frameData })
});
const result = await analysis.json();
console.log('Eye Contact:', result.result.metrics.eye_contact);

// 3. End Session & Get Report
const report = await fetch(`http://localhost:5001/api/sessions/${session_id}/end`, {
    method: 'POST'
});
const finalReport = await report.json();
'''
        }
    })


# ==================== COMPREHENSIVE INTERVIEW ANALYSIS (Video URL based) ====================

@app.route('/comprehensive-interview-analysis', methods=['POST', 'OPTIONS'])
def comprehensive_interview_analysis():
    """
    Comprehensive interview analysis endpoint for pre-recorded videos.
    Accepts video URL, analyzes frames, and returns detailed report.
    """
    if request.method == 'OPTIONS':
        return make_response(), 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        # Get parameters
        video_url = data.get('video_url')
        questions_with_answers = data.get('questionsWithAnswer', [])
        job_data = data.get('jobData', {})
        frame_interval = data.get('frame_interval', 60)
        
        if not video_url:
            return jsonify({
                'success': False,
                'error': 'video_url is required'
            }), 400
        
        logger.info(f"Starting comprehensive analysis for video: {video_url}")
        
        # Create a session for analysis
        session_id = str(uuid.uuid4())
        session = LiveInterviewSession(session_id, job_data)
        
        # Download and analyze video
        temp_video_path = None
        try:
            # Download video
            temp_video_path = download_video(video_url)
            
            # Extract and analyze frames
            analyze_video_frames(session, temp_video_path, frame_interval)
            
            # Generate final report
            report = session.generate_final_report()
            
            # Add Q&A analysis if provided
            if questions_with_answers:
                qa_analysis = analyze_questions(questions_with_answers, job_data)
                report['qa_analysis'] = qa_analysis
            
            return jsonify({
                'success': True,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'video_url': video_url,
                'job_title': job_data.get('jobTitle', 'Not specified'),
                **report
            })
            
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


def download_video(video_url: str) -> str:
    """Download video from URL to temporary file"""
    logger.info(f"Downloading video from: {video_url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(video_url, stream=True, timeout=120, headers=headers)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '')
    file_extension = '.mp4'
    if 'webm' in content_type:
        file_extension = '.webm'
    elif 'avi' in content_type:
        file_extension = '.avi'
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            temp_file.write(chunk)
    
    temp_file.close()
    logger.info(f"Video downloaded to: {temp_file.name}")
    return temp_file.name


def analyze_video_frames(session: 'LiveInterviewSession', video_path: str, frame_interval: int = 60):
    """Extract frames from video and analyze each"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {total_frames} frames, {fps} fps, analyzing every {frame_interval} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Analyze frame
            session.analyze_frame(frame_bytes)
            logger.info(f"Analyzed frame {frame_count}/{total_frames}")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Video analysis complete. Processed {session.analyzed_frame_count} frames.")


def analyze_questions(questions_with_answers: list, job_data: dict) -> dict:
    """Analyze Q&A data"""
    total_questions = len(questions_with_answers)
    answered = [q for q in questions_with_answers if q.get('score', 0) > 0]
    
    # Calculate scores by category
    category_scores = {}
    for q in questions_with_answers:
        category = q.get('questionDetails', {}).get('category', 'General')
        score = q.get('score', 0)
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
    
    category_averages = {}
    for cat, scores in category_scores.items():
        if scores:
            category_averages[cat] = round(sum(scores) / len(scores) * 20, 1)
    
    overall_score = sum(q.get('score', 0) for q in questions_with_answers) / max(total_questions, 1) * 20
    
    return {
        'total_questions': total_questions,
        'answered_questions': len(answered),
        'overall_score_percentage': round(overall_score, 1),
        'category_averages': category_averages,
        'completion_rate': round(len(answered) / max(total_questions, 1) * 100, 1)
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Live Interview Analyzer API',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'aws_rekognition': AWS_ACCESS_KEY_ID is not None,
            'openai_llm': OPENAI_API_KEY is not None,
            'anthropic_llm': ANTHROPIC_API_KEY is not None,
            'websocket_enabled': True,
            'rest_api_enabled': True,
            'video_url_analysis': True
        },
        'active_sessions': len([s for s in active_sessions.values() if s.is_active])
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Live Interview Analyzer API',
        'version': '1.0.0',
        'description': 'Real-time AI-powered interview video analysis',
        'endpoints': {
            'WebSocket Events': {
                'connect': 'Establish WebSocket connection',
                'start_session': 'Start a new live analysis session',
                'analyze_frame': 'Send a frame for real-time analysis',
                'get_live_metrics': 'Get current session metrics',
                'end_session': 'End session and get final report'
            },
            'REST API': {
                'POST /api/sessions/start': 'Start a new session',
                'POST /api/sessions/{id}/frame': 'Analyze a frame',
                'GET /api/sessions/{id}/metrics': 'Get session metrics',
                'POST /api/sessions/{id}/end': 'End session',
                'GET /api/sessions': 'List all sessions',
                'GET /health': 'Health check'
            }
        }
    })


if __name__ == '__main__':
    logger.info("🚀 Live Interview Analyzer API v1.0.0 starting...")
    logger.info("✅ WebSocket support enabled (Flask-SocketIO)")
    logger.info("✅ REST API endpoints available")
    logger.info("✅ Real-time frame analysis ready")
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.warning("⚠️  AWS credentials missing - add to environment variables")
    else:
        logger.info("✅ AWS Rekognition configured")
    
    if OPENAI_API_KEY:
        logger.info("✅ OpenAI API configured")
    elif ANTHROPIC_API_KEY:
        logger.info("✅ Anthropic API configured")
    else:
        logger.warning("⚠️  No LLM API keys configured")
    
    logger.info("🌍 API starting at: http://localhost:5001")
    logger.info("🔌 WebSocket endpoint: ws://localhost:5001/socket.io/")
    logger.info("📡 Ready to analyze live video sessions!")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)

