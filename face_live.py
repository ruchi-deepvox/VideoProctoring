from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import boto3
import json
import cv2
import numpy as np
import base64
import os
from datetime import datetime, timedelta
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
        
        # Face verification
        self.reference_face_image: bytes = None  # Stored reference face image
        self.face_registered = False
        self.face_verification_enabled = True
        self.face_match_scores: List[float] = []  # Track match scores over time
        self.face_mismatch_counter = 0
        self.face_mismatch_threshold = 3  # Alert after 3 consecutive mismatches
        self.face_similarity_threshold = 80.0  # Minimum similarity percentage to consider a match
        self.verification_results: List[Dict] = []  # Store all verification results
        
        # Document verification
        self.document_verified = False
        self.document_type: str = None  # 'aadhaar', 'pan', 'passport', 'driving_license', 'other'
        self.document_face_image: bytes = None  # Face extracted from document
        self.document_verification_result: Dict = None
        self.document_similarity_threshold = 80.0  # Minimum similarity for document match
        
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
    
    def register_face(self, face_image: bytes) -> Dict:
        """Register the candidate's reference face for verification"""
        if not self.rekognition:
            return {'success': False, 'error': 'AWS Rekognition not initialized'}
        
        try:
            # Detect face in the reference image
            face_response = self.rekognition.detect_faces(
                Image={'Bytes': face_image},
                Attributes=['DEFAULT']
            )
            
            faces = face_response.get('FaceDetails', [])
            
            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'No face detected in the image. Please ensure your face is clearly visible.'
                }
            
            if len(faces) > 1:
                return {
                    'success': False,
                    'error': 'Multiple faces detected. Please ensure only you are in the frame.'
                }
            
            # Check face quality
            face = faces[0]
            confidence = face.get('Confidence', 0)
            
            if confidence < 90:
                return {
                    'success': False,
                    'error': f'Face detection confidence too low ({confidence:.1f}%). Please ensure good lighting and face the camera directly.'
                }
            
            # Store the reference face
            self.reference_face_image = face_image
            self.face_registered = True
            
            # Get face bounding box info
            bbox = face.get('BoundingBox', {})
            
            logger.info(f"Session {self.session_id}: Face registered successfully with {confidence:.1f}% confidence")
            
            return {
                'success': True,
                'message': 'Face registered successfully',
                'face_confidence': confidence,
                'face_bounding_box': bbox,
                'verification_enabled': self.face_verification_enabled
            }
            
        except Exception as e:
            logger.error(f"Session {self.session_id}: Face registration failed: {e}")
            return {
                'success': False,
                'error': f'Face registration failed: {str(e)}'
            }
    
    def verify_document(self, document_image: bytes, live_face_image: bytes, document_type: str = 'other') -> Dict:
        """
        Verify candidate's identity by comparing face from ID document with live face.
        
        Args:
            document_image: Image of the ID document (Aadhaar, PAN, Passport, etc.)
            live_face_image: Live captured face image of the candidate
            document_type: Type of document ('aadhaar', 'pan', 'passport', 'driving_license', 'voter_id', 'other')
        
        Returns:
            Dict with verification result
        """
        if not self.rekognition:
            return {'success': False, 'error': 'AWS Rekognition not initialized'}
        
        try:
            # Step 1: Detect face in the document
            logger.info(f"Session {self.session_id}: Detecting face in {document_type} document...")
            
            doc_face_response = self.rekognition.detect_faces(
                Image={'Bytes': document_image},
                Attributes=['DEFAULT']
            )
            
            doc_faces = doc_face_response.get('FaceDetails', [])
            
            if len(doc_faces) == 0:
                return {
                    'success': False,
                    'verified': False,
                    'error': 'No face detected in the document. Please upload a clear image of your ID with a visible photo.',
                    'document_type': document_type
                }
            
            # Get document face confidence
            doc_face = doc_faces[0]
            doc_face_confidence = doc_face.get('Confidence', 0)
            doc_face_bbox = doc_face.get('BoundingBox', {})
            
            logger.info(f"Session {self.session_id}: Document face detected with {doc_face_confidence:.1f}% confidence")
            
            # Step 2: Detect face in the live image
            live_face_response = self.rekognition.detect_faces(
                Image={'Bytes': live_face_image},
                Attributes=['DEFAULT']
            )
            
            live_faces = live_face_response.get('FaceDetails', [])
            
            if len(live_faces) == 0:
                return {
                    'success': False,
                    'verified': False,
                    'error': 'No face detected in the live image. Please ensure your face is clearly visible.',
                    'document_type': document_type
                }
            
            if len(live_faces) > 1:
                return {
                    'success': False,
                    'verified': False,
                    'error': 'Multiple faces detected in live image. Please ensure only you are in the frame.',
                    'document_type': document_type
                }
            
            live_face = live_faces[0]
            live_face_confidence = live_face.get('Confidence', 0)
            
            logger.info(f"Session {self.session_id}: Live face detected with {live_face_confidence:.1f}% confidence")
            
            # Step 3: Compare faces using AWS Rekognition
            try:
                compare_response = self.rekognition.compare_faces(
                    SourceImage={'Bytes': document_image},
                    TargetImage={'Bytes': live_face_image},
                    SimilarityThreshold=50.0  # Low threshold to get comparison data
                )
                
                face_matches = compare_response.get('FaceMatches', [])
                unmatched_faces = compare_response.get('UnmatchedFaces', [])
                
                if face_matches:
                    # Get the best match
                    best_match = max(face_matches, key=lambda x: x.get('Similarity', 0))
                    similarity = best_match.get('Similarity', 0)
                    
                    is_verified = similarity >= self.document_similarity_threshold
                    
                    # Store verification result
                    self.document_verified = is_verified
                    self.document_type = document_type
                    self.document_face_image = document_image
                    self.document_verification_result = {
                        'verified': is_verified,
                        'similarity': round(similarity, 2),
                        'document_type': document_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Also register this as the reference face if verification passed
                    if is_verified and not self.face_registered:
                        self.reference_face_image = live_face_image
                        self.face_registered = True
                    
                    logger.info(f"Session {self.session_id}: Document verification {'PASSED' if is_verified else 'FAILED'} with {similarity:.1f}% similarity")
                    
                    result = {
                        'success': True,
                        'verified': is_verified,
                        'similarity': round(similarity, 2),
                        'threshold': self.document_similarity_threshold,
                        'document_type': document_type,
                        'document_face_confidence': round(doc_face_confidence, 2),
                        'live_face_confidence': round(live_face_confidence, 2),
                        'message': 'Identity verified successfully! Document face matches live face.' if is_verified else f'Identity verification failed. Face similarity ({similarity:.1f}%) is below threshold ({self.document_similarity_threshold}%).',
                        'face_registered': self.face_registered
                    }
                    
                    if is_verified:
                        result['next_step'] = 'You can now proceed with the interview.'
                    else:
                        result['next_step'] = 'Please try again with a clearer document or better lighting.'
                    
                    return result
                    
                else:
                    # No match found
                    self.document_verified = False
                    self.document_verification_result = {
                        'verified': False,
                        'similarity': 0,
                        'document_type': document_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.warning(f"Session {self.session_id}: Document verification FAILED - faces do not match")
                    
                    return {
                        'success': True,
                        'verified': False,
                        'similarity': 0,
                        'threshold': self.document_similarity_threshold,
                        'document_type': document_type,
                        'document_face_confidence': round(doc_face_confidence, 2),
                        'live_face_confidence': round(live_face_confidence, 2),
                        'message': 'Identity verification failed. The face in the document does not match your live face.',
                        'next_step': 'Please ensure you are using your own ID document and try again.'
                    }
                    
            except self.rekognition.exceptions.InvalidParameterException as e:
                logger.error(f"Session {self.session_id}: Face comparison failed: {e}")
                return {
                    'success': False,
                    'verified': False,
                    'error': 'Could not compare faces. Please ensure both images have clear, visible faces.',
                    'document_type': document_type
                }
                
        except Exception as e:
            logger.error(f"Session {self.session_id}: Document verification error: {e}")
            return {
                'success': False,
                'verified': False,
                'error': f'Document verification failed: {str(e)}',
                'document_type': document_type
            }
    
    def get_document_verification_status(self) -> Dict:
        """Get the current document verification status"""
        return {
            'document_verified': self.document_verified,
            'document_type': self.document_type,
            'verification_result': self.document_verification_result,
            'face_registered': self.face_registered
        }
    
    def verify_face(self, frame_image: bytes) -> Dict:
        """Compare current frame face with registered reference face"""
        if not self.face_registered or not self.reference_face_image:
            return {
                'verified': None,
                'message': 'No reference face registered',
                'similarity': 0
            }
        
        if not self.rekognition:
            return {
                'verified': None,
                'message': 'AWS Rekognition not initialized',
                'similarity': 0
            }
        
        try:
            # Compare faces using AWS Rekognition
            compare_response = self.rekognition.compare_faces(
                SourceImage={'Bytes': self.reference_face_image},
                TargetImage={'Bytes': frame_image},
                SimilarityThreshold=50.0  # Low threshold to get comparison even for different people
            )
            
            face_matches = compare_response.get('FaceMatches', [])
            unmatched_faces = compare_response.get('UnmatchedFaces', [])
            
            if face_matches:
                # Face matched - get similarity score
                best_match = max(face_matches, key=lambda x: x.get('Similarity', 0))
                similarity = best_match.get('Similarity', 0)
                
                is_match = similarity >= self.face_similarity_threshold
                
                # Track match scores
                self.face_match_scores.append(similarity)
                
                if is_match:
                    self.face_mismatch_counter = 0  # Reset mismatch counter
                else:
                    self.face_mismatch_counter += 1
                
                result = {
                    'verified': is_match,
                    'similarity': round(similarity, 2),
                    'message': 'Face verified' if is_match else f'Face similarity too low ({similarity:.1f}%)',
                    'threshold': self.face_similarity_threshold
                }
                
            elif unmatched_faces:
                # Different person detected
                self.face_mismatch_counter += 1
                self.face_match_scores.append(0)
                
                result = {
                    'verified': False,
                    'similarity': 0,
                    'message': 'Different person detected',
                    'threshold': self.face_similarity_threshold
                }
            else:
                # No face found in current frame
                result = {
                    'verified': None,
                    'similarity': 0,
                    'message': 'No face detected in current frame',
                    'threshold': self.face_similarity_threshold
                }
            
            # Store verification result
            self.verification_results.append({
                'timestamp': (datetime.now() - self.start_time).total_seconds(),
                **result
            })
            
            # Check if we need to generate an alert
            if self.face_mismatch_counter >= self.face_mismatch_threshold:
                self._add_alert('face_mismatch', f'Face verification failed {self.face_mismatch_counter} consecutive times')
            
            return result
            
        except self.rekognition.exceptions.InvalidParameterException as e:
            # This usually means no face in one of the images
            return {
                'verified': None,
                'similarity': 0,
                'message': 'Could not compare faces - ensure face is visible',
                'threshold': self.face_similarity_threshold
            }
        except Exception as e:
            logger.error(f"Session {self.session_id}: Face verification error: {e}")
            return {
                'verified': None,
                'similarity': 0,
                'message': f'Verification error: {str(e)}',
                'threshold': self.face_similarity_threshold
            }
    
    def get_face_verification_summary(self) -> Dict:
        """Get summary of face verification during the session"""
        if not self.face_match_scores:
            return {
                'enabled': self.face_verification_enabled,
                'registered': self.face_registered,
                'total_verifications': 0,
                'average_similarity': 0,
                'verification_rate': 0,
                'mismatches': 0
            }
        
        successful_verifications = len([s for s in self.face_match_scores if s >= self.face_similarity_threshold])
        
        return {
            'enabled': self.face_verification_enabled,
            'registered': self.face_registered,
            'total_verifications': len(self.face_match_scores),
            'successful_verifications': successful_verifications,
            'average_similarity': round(sum(self.face_match_scores) / len(self.face_match_scores), 2),
            'max_similarity': round(max(self.face_match_scores), 2),
            'min_similarity': round(min(self.face_match_scores), 2),
            'verification_rate': round((successful_verifications / len(self.face_match_scores)) * 100, 1),
            'total_mismatches': len([s for s in self.face_match_scores if s < self.face_similarity_threshold])
        }
    
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
            
            # Perform face verification if enabled and registered
            if self.face_verification_enabled and self.face_registered:
                verification_result = self.verify_face(frame_data)
                analysis_result['face_verification'] = verification_result
            else:
                analysis_result['face_verification'] = {
                    'verified': None,
                    'message': 'Face not registered' if not self.face_registered else 'Verification disabled',
                    'similarity': 0
                }
            
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
        """Generate comprehensive final report in the required UI format"""
        duration = (self.end_time or datetime.now()) - self.start_time
        
        # Calculate behavioral metrics
        behavioral_metrics = self._calculate_behavioral_metrics()
        body_language_analysis = self._analyze_body_language()
        cultural_fit_analysis = self._analyze_cultural_fit()
        video_insights = self._generate_video_insights(behavioral_metrics, body_language_analysis)
        
        # Generate LLM-enhanced summary if available
        ai_summary = self._generate_ai_summary(behavioral_metrics, body_language_analysis)
        
        # Calculate performance scores
        behavior_avg = round((behavioral_metrics['confidence'] + behavioral_metrics['engagement'] + 
                             behavioral_metrics['eye_contact'] + behavioral_metrics['posture'] + 
                             behavioral_metrics['gestures'] + behavioral_metrics['voice_tone']) / 6)
        
        body_language_avg = body_language_analysis.get('overall_score', 80)
        cultural_fit_avg = cultural_fit_analysis.get('overall_score', 85)
        
        # Build the response in required format
        return {
            'status': 'success',
            'session_id': self.session_id,
            
            # Behavioral Analysis (required format)
            'behavioral_analysis': {
                'confidence': behavioral_metrics['confidence'],
                'engagement': behavioral_metrics['engagement'],
                'eye_contact': behavioral_metrics['eye_contact'],
                'facial_expressions': behavioral_metrics['facial_expressions'],
                'gestures': behavioral_metrics['gestures'],
                'posture': behavioral_metrics['posture'],
                'voice_tone': behavioral_metrics['voice_tone']
            },
            
            # Performance Breakdown (required format)
            'performanceBreakdown': {
                'behavior': {
                    'overallAveragePercentage': behavior_avg,
                    'summary': self._get_performance_summary('behavior', behavior_avg)
                },
                'body_language': {
                    'overallAveragePercentage': body_language_avg,
                    'summary': self._get_performance_summary('body_language', body_language_avg)
                },
                'communicationSkills': {
                    'answeredAveragePercentage': round((behavioral_metrics['engagement'] + behavioral_metrics['facial_expressions']) / 2),
                    'overallAveragePercentage': round((behavioral_metrics['engagement'] + behavioral_metrics['facial_expressions']) / 2),
                    'summary': self._get_performance_summary('communication', round((behavioral_metrics['engagement'] + behavioral_metrics['facial_expressions']) / 2))
                },
                'confidenceLevel': {
                    'answeredAveragePercentage': behavioral_metrics['confidence'],
                    'overallAveragePercentage': behavioral_metrics['confidence'],
                    'summary': self._get_performance_summary('confidence', behavioral_metrics['confidence'])
                },
                'culturalFit': {
                    'overallAveragePercentage': cultural_fit_avg,
                    'summary': self._get_performance_summary('cultural_fit', cultural_fit_avg)
                },
                'leadershipPotential': {
                    'answeredAveragePercentage': round(behavioral_metrics['confidence'] * 0.7),
                    'overallAveragePercentage': round(behavioral_metrics['confidence'] * 0.7),
                    'summary': self._get_performance_summary('leadership', round(behavioral_metrics['confidence'] * 0.7))
                },
                'problemSolving': {
                    'answeredAveragePercentage': round((behavioral_metrics['engagement'] + behavioral_metrics['eye_contact']) / 2 * 0.8),
                    'overallAveragePercentage': round((behavioral_metrics['engagement'] + behavioral_metrics['eye_contact']) / 2 * 0.8),
                    'summary': self._get_performance_summary('problem_solving', round((behavioral_metrics['engagement'] + behavioral_metrics['eye_contact']) / 2 * 0.8))
                },
                'technicalKnowledge': {
                    'answeredAveragePercentage': round(behavioral_metrics['engagement'] * 0.6),
                    'overallAveragePercentage': round(behavioral_metrics['engagement'] * 0.6),
                    'summary': self._get_performance_summary('technical', round(behavioral_metrics['engagement'] * 0.6))
                },
                'professionalAttire': {
                    'overallAveragePercentage': cultural_fit_avg,
                    'summary': self._get_performance_summary('cultural_fit', cultural_fit_avg)
                }
            },
            
            # Quick Stats (required format)
            'quickStats': {
                'communicationSkills': self._get_level_label(round((behavioral_metrics['engagement'] + behavioral_metrics['facial_expressions']) / 2)),
                'confidenceLevel': self._get_level_label(behavioral_metrics['confidence']),
                'leadershipPotential': self._get_level_label(round(behavioral_metrics['confidence'] * 0.7)),
                'problemSolving': self._get_level_label(round((behavioral_metrics['engagement'] + behavioral_metrics['eye_contact']) / 2 * 0.8)),
                'technicalKnowledge': self._get_level_label(round(behavioral_metrics['engagement'] * 0.6)),
                'professionalAttire': self._get_level_label(cultural_fit_avg)
            },
            
            # AI Evaluation Summary - USE LLM when available for dynamic, varied feedback
            'aiEvaluationSummary': self._build_ai_evaluation_summary(behavioral_metrics, body_language_analysis, ai_summary, video_insights),
            
            # Video Analysis Insights (required format)
            'video_analysis_insights': {
                'keyStrengths': video_insights.get('positive_indicators', []) or video_insights.get('keyStrengths', []),
                'areasOfGrowth': video_insights.get('areas_for_improvement', []),
                'positive_indicators': video_insights.get('positive_indicators', []),
                'areas_for_improvement': video_insights.get('areas_for_improvement', []),
                'recommendations': video_insights.get('recommendations', [])
            },
            
            # Recommendations (required format)
            'recommendations': self._generate_final_recommendation(behavior_avg, body_language_avg, cultural_fit_avg),
            
            # Additional data for reference
            'session_metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': (self.end_time or datetime.now()).isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_frames_analyzed': self.analyzed_frame_count,
                'total_alerts': len(self.alert_history),
                'document_verified': self.document_verified,
                'face_registered': self.face_registered
            },
            'document_verification': self.get_document_verification_status(),
            'face_verification_summary': self.get_face_verification_summary(),
            'emotion_summary': behavioral_metrics.get('emotion_summary', {}),
            'alert_history': self.alert_history,
            'token_consumption': self.token_tracker.get_summary()
        }
    
    def _get_level_label(self, score: int) -> str:
        """Convert numeric score to text label"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _get_performance_summary(self, category: str, score: int) -> str:
        """Generate summary text for each performance category"""
        summaries = {
            'behavior': {
                90: "Outstanding interview performance with exceptional behavioral presence and communication.",
                75: "Strong interview performance with excellent behavioral presence and communication.",
                60: "Good interview performance with adequate behavioral presence.",
                40: "Fair interview performance with room for improvement in behavioral aspects.",
                0: "Interview performance needs significant improvement in behavioral areas."
            },
            'body_language': {
                90: "Excellent body language with confident posture and natural gestures.",
                75: "Good body language with appropriate posture and gestures, showing professional presence.",
                60: "Adequate body language, though some nervousness may be apparent.",
                40: "Body language shows signs of nervousness or discomfort.",
                0: "Body language needs significant improvement for professional settings."
            },
            'communication': {
                90: "Excellent communication skills with clear and engaging delivery.",
                75: "Good communication skills demonstrating clarity and engagement.",
                60: "The candidate demonstrates a fair understanding of communication in a technical context.",
                40: "Communication skills need improvement for clearer expression.",
                0: "Communication skills require significant development."
            },
            'confidence': {
                90: "The candidate's confidence appears excellent based on behavioral analysis.",
                75: "Good confidence level demonstrated throughout the interview.",
                60: "Moderate confidence level with some hesitation observed.",
                40: "Confidence level needs improvement.",
                0: "Low confidence observed, may benefit from interview practice."
            },
            'cultural_fit': {
                90: "Excellent cultural fit with professional attire and presentation.",
                75: "Good cultural fit with appropriate professional demeanor.",
                60: "Adequate cultural fit with room for improvement.",
                40: "Some concerns about cultural fit observed.",
                0: "Cultural fit assessment indicates potential challenges."
            },
            'leadership': {
                90: "Strong leadership potential demonstrated through confident presence.",
                75: "Good leadership qualities observed in communication style.",
                60: "Some leadership potential evident, could be developed further.",
                40: "Leadership qualities were not strongly demonstrated.",
                0: "No leadership questions were answered adequately."
            },
            'problem_solving': {
                90: "Excellent analytical approach and problem-solving methodology.",
                75: "Good problem-solving skills with structured thinking.",
                60: "Adequate problem-solving approach demonstrated.",
                40: "Problem-solving skills need further development.",
                0: "The candidate has a poor approach to problem-solving."
            },
            'technical': {
                90: "Excellent technical knowledge demonstrated.",
                75: "Good level of technical knowledge shown.",
                60: "Fair technical knowledge with room for deeper understanding.",
                40: "Technical knowledge needs strengthening.",
                0: "The candidate shows a poor level of technical knowledge but may need deeper explanations."
            }
        }
        
        category_summaries = summaries.get(category, summaries['behavior'])
        
        for threshold in sorted(category_summaries.keys(), reverse=True):
            if score >= threshold:
                return category_summaries[threshold]
        
        return category_summaries[0]
    
    def _build_ai_evaluation_summary(self, behavioral: Dict, body_language: Dict, ai_summary: Optional[Dict], video_insights: Dict) -> Dict:
        """
        Build AI evaluation summary - USE LLM output when available for dynamic, varied feedback.
        Falls back to rule-based when LLM unavailable.
        """
        # Use LLM-generated summary when available - produces varied, contextual feedback per session
        if ai_summary and isinstance(ai_summary, dict):
            llm_summary = ai_summary.get('summary', '').strip()
            llm_strengths = ai_summary.get('keyStrengths') or ai_summary.get('strengths')
            llm_growth = ai_summary.get('areasOfGrowth') or ai_summary.get('areas_of_growth')
            
            if llm_summary and (llm_strengths or llm_growth):
                return {
                    'summary': llm_summary,
                    'keyStrengths': (llm_strengths if isinstance(llm_strengths, list) else [llm_strengths])[:5] if llm_strengths else [],
                    'areasOfGrowth': (llm_growth if isinstance(llm_growth, list) else [llm_growth])[:5] if llm_growth else []
                }
        
        # Fallback: rule-based (when LLM unavailable or returned invalid data)
        return self._generate_ai_evaluation_summary_fallback(behavioral, body_language, video_insights)
    
    def _generate_ai_evaluation_summary_fallback(self, behavioral: Dict, body_language: Dict, video_insights: Dict) -> Dict:
        """Fallback rule-based AI evaluation when LLM unavailable"""
        avg_score = (behavioral.get('confidence', 0) + behavioral.get('engagement', 0) + behavioral.get('eye_contact', 0)) / 3
        
        # Merge video insights with rule-based - ensures variation from session data
        key_strengths = list(video_insights.get('positive_indicators', []))[:2]
        areas_of_growth = list(video_insights.get('areas_for_improvement', []))[:2]
        
        if behavioral.get('confidence', 0) >= 80 and not any('confidence' in s.lower() for s in key_strengths):
            key_strengths.insert(0, "You demonstrate strong self-assurance and confidence throughout the interview, which creates a positive impression.")
        if behavioral.get('eye_contact', 0) >= 80 and not any('eye contact' in s.lower() for s in key_strengths):
            key_strengths.insert(0, "Excellent eye contact maintained consistently, showing engagement and building rapport effectively.")
        if behavioral.get('posture', 0) >= 80 and not any('posture' in s.lower() for s in key_strengths):
            key_strengths.append("Professional posture maintained throughout, projecting confidence and readiness.")
        
        if behavioral.get('facial_expressions', 100) < 70:
            areas_of_growth.insert(0, "Practice using more varied facial expressions to convey enthusiasm and engagement during responses.")
        if behavioral.get('engagement', 100) < 70:
            areas_of_growth.append("Focus on staying more engaged and attentive throughout longer interview sessions.")
        if behavioral.get('eye_contact', 100) < 70:
            areas_of_growth.append("Maintain more consistent eye contact to build better rapport with interviewers.")
        
        if not key_strengths:
            key_strengths = ["Shows willingness to participate and engage.", "Demonstrates basic understanding of concepts."]
        if not areas_of_growth:
            areas_of_growth = ["Continue to refine communication skills.", "Provide more specific examples when answering."]
        
        if avg_score >= 85:
            summary = "Excellent interview performance! You demonstrated strong confidence, engagement, and professional presence throughout."
        elif avg_score >= 70:
            summary = "Good interview performance with solid behavioral indicators. Some areas show room for improvement, but overall presentation was professional."
        elif avg_score >= 55:
            summary = "The interview revealed some areas where you shine and others where there's room for growth. Focus on the areas identified for improvement."
        else:
            summary = "The interview indicates several areas that need development. Practice and preparation will help improve your interview performance."
        
        return {
            'keyStrengths': key_strengths[:5],
            'areasOfGrowth': areas_of_growth[:5],
            'summary': summary
        }
    
    def _generate_final_recommendation(self, behavior_avg: int, body_language_avg: int, cultural_fit_avg: int) -> Dict:
        """Generate final hiring recommendation"""
        overall_avg = (behavior_avg + body_language_avg + cultural_fit_avg) / 3
        
        if overall_avg >= 85:
            return {
                'recommendation': 'Strongly recommend',
                'summary': 'Excellent candidate with strong behavioral indicators and professional presence. Highly recommended for the position.'
            }
        elif overall_avg >= 70:
            return {
                'recommendation': 'Recommend',
                'summary': 'Good candidate with solid performance. Recommend proceeding to the next stage of the hiring process.'
            }
        elif overall_avg >= 55:
            return {
                'recommendation': 'Consider with reservations',
                'summary': 'Candidate has potential but may need additional training or support in key areas.'
            }
        else:
            return {
                'recommendation': 'Not recommended at this time',
                'summary': 'Candidate may not be ready for this role. Consider for future opportunities after skill development.'
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
        
        # Define alert_types BEFORE using it in the prompt
        alert_types = [a['type'] for a in self.alert_history]
        alert_summary = {}
        for alert_type in alert_types:
            alert_summary[alert_type] = alert_types.count(alert_type)
        
        emotion_data = behavioral_analysis.get('emotion_summary', {})
        emotion_str = ', '.join(f'{k}: {round(v)}%' for k, v in emotion_data.items()) if emotion_data else 'Not available'
        
        prompt = f"""
        Analyze this SPECIFIC interview session and provide a UNIQUE, personalized evaluation. Each session is different - your feedback must reflect THIS candidate's actual metrics.
        
        SESSION DATA (use these exact numbers - they vary per interview):
        - Duration: {((self.end_time or datetime.now()) - self.start_time).total_seconds():.0f} seconds
        - Frames Analyzed: {self.analyzed_frame_count}
        - Alerts During Session: {len(self.alert_history)} - Types: {list(set(alert_types)) if alert_types else 'None'}
        
        BEHAVIORAL METRICS (candidate-specific, use for personalized feedback):
        - Eye Contact: {behavioral_analysis.get('eye_contact', 0)}%
        - Confidence: {behavioral_analysis.get('confidence', 0)}%
        - Engagement: {behavioral_analysis.get('engagement', 0)}%
        - Posture: {behavioral_analysis.get('posture', 0)}%
        - Facial Expressions: {behavioral_analysis.get('facial_expressions', 0)}%
        - Gestures: {behavioral_analysis.get('gestures', 0)}%
        - Voice Tone: {behavioral_analysis.get('voice_tone', 0)}%
        
        Body Language Score: {body_language_analysis.get('overall_score', 0)}%
        Emotion Distribution: {emotion_str}
        
        IMPORTANT: Write SPECIFIC feedback based on these exact numbers. A 92% eye contact deserves different praise than 65%. A 41% facial expression needs specific improvement advice. Do NOT give generic feedback - make it unique to THIS session's data.
        
        Respond in valid JSON only (overallVideoScore must be a number 0-100):
        {{
            "summary": "2-3 sentences tailored to these specific scores",
            "keyStrengths": ["specific strength based on high scores", "another specific strength"],
            "areasOfGrowth": ["specific improvement for low scores", "another specific area"],
            "overallVideoScore": 75
        }}
        """
        
        try:
            if self.llm_provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert interview coach. Provide unique, personalized feedback for each candidate. Never repeat the same phrases - tailor every response to the specific metrics provided."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.85,
                    max_tokens=600
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

# Store verified documents temporarily (before session creation)
verified_documents = {}  # verification_token -> verification_data
verification_lock = threading.Lock()


@app.route('/api/verify-document', methods=['POST'])
def api_verify_document_standalone():
    """
    Standalone document verification - verify identity BEFORE creating a session.
    Returns a verification token that can be used when starting a session.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Get document image
        if 'document' not in data:
            return jsonify({'success': False, 'error': 'No document image provided. Please upload your ID document.'}), 400
        
        # Get live face image
        if 'live_face' not in data:
            return jsonify({'success': False, 'error': 'No live face image provided. Please capture your face.'}), 400
        
        # Get document type (optional)
        document_type = data.get('document_type', 'other')
        valid_document_types = ['aadhaar', 'pan', 'passport', 'driving_license', 'voter_id', 'other']
        if document_type not in valid_document_types:
            document_type = 'other'
        
        # Get candidate info (optional)
        candidate_name = data.get('candidate_name', '')
        
        # Decode document image
        document_data = data['document']
        if ',' in document_data:
            document_data = document_data.split(',')[1]
        document_bytes = base64.b64decode(document_data)
        
        # Decode live face image
        live_face_data = data['live_face']
        if ',' in live_face_data:
            live_face_data = live_face_data.split(',')[1]
        live_face_bytes = base64.b64decode(live_face_data)
        
        # Initialize Rekognition client
        try:
            rekognition = boto3.client(
                'rekognition',
                region_name='us-east-1',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize AWS Rekognition: {e}")
            return jsonify({'success': False, 'error': 'AWS service unavailable'}), 500
        
        # Step 1: Detect face in the document
        logger.info(f"Standalone verification: Detecting face in {document_type} document...")
        
        doc_face_response = rekognition.detect_faces(
            Image={'Bytes': document_bytes},
            Attributes=['DEFAULT']
        )
        
        doc_faces = doc_face_response.get('FaceDetails', [])
        
        if len(doc_faces) == 0:
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'No face detected in the document. Please upload a clear image of your ID with a visible photo.',
                'document_type': document_type
            }), 400
        
        doc_face = doc_faces[0]
        doc_face_confidence = doc_face.get('Confidence', 0)
        
        # Step 2: Detect face in the live image
        live_face_response = rekognition.detect_faces(
            Image={'Bytes': live_face_bytes},
            Attributes=['DEFAULT']
        )
        
        live_faces = live_face_response.get('FaceDetails', [])
        
        if len(live_faces) == 0:
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'No face detected in the live image. Please ensure your face is clearly visible.',
                'document_type': document_type
            }), 400
        
        if len(live_faces) > 1:
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'Multiple faces detected. Please ensure only you are in the frame.',
                'document_type': document_type
            }), 400
        
        live_face = live_faces[0]
        live_face_confidence = live_face.get('Confidence', 0)
        
        # Step 3: Compare faces
        similarity_threshold = 80.0
        
        try:
            compare_response = rekognition.compare_faces(
                SourceImage={'Bytes': document_bytes},
                TargetImage={'Bytes': live_face_bytes},
                SimilarityThreshold=50.0
            )
            
            face_matches = compare_response.get('FaceMatches', [])
            
            if face_matches:
                best_match = max(face_matches, key=lambda x: x.get('Similarity', 0))
                similarity = best_match.get('Similarity', 0)
                is_verified = similarity >= similarity_threshold
                
                if is_verified:
                    # Generate verification token
                    verification_token = str(uuid.uuid4())
                    
                    # Store verification data (expires in 30 minutes)
                    with verification_lock:
                        verified_documents[verification_token] = {
                            'verified': True,
                            'similarity': round(similarity, 2),
                            'document_type': document_type,
                            'candidate_name': candidate_name,
                            'live_face_image': live_face_bytes,  # Store for session
                            'timestamp': datetime.now().isoformat(),
                            'expires_at': (datetime.now() + timedelta(minutes=30)).isoformat()
                        }
                    
                    logger.info(f"Document verification PASSED: {similarity:.1f}% similarity, token: {verification_token[:8]}...")
                    
                    return jsonify({
                        'success': True,
                        'verified': True,
                        'verification_token': verification_token,
                        'similarity': round(similarity, 2),
                        'threshold': similarity_threshold,
                        'document_type': document_type,
                        'document_face_confidence': round(doc_face_confidence, 2),
                        'live_face_confidence': round(live_face_confidence, 2),
                        'message': 'Identity verified successfully! You can now proceed to start the interview.',
                        'next_step': 'Use the verification_token when calling /api/sessions/start',
                        'token_expires_in': '30 minutes'
                    })
                else:
                    logger.warning(f"Document verification FAILED: {similarity:.1f}% similarity (threshold: {similarity_threshold}%)")
                    
                    return jsonify({
                        'success': True,
                        'verified': False,
                        'similarity': round(similarity, 2),
                        'threshold': similarity_threshold,
                        'document_type': document_type,
                        'document_face_confidence': round(doc_face_confidence, 2),
                        'live_face_confidence': round(live_face_confidence, 2),
                        'message': f'Identity verification failed. Face similarity ({similarity:.1f}%) is below threshold ({similarity_threshold}%).',
                        'next_step': 'Please try again with a clearer document or better lighting.'
                    })
            else:
                logger.warning("Document verification FAILED: No face match found")
                
                return jsonify({
                    'success': True,
                    'verified': False,
                    'similarity': 0,
                    'threshold': similarity_threshold,
                    'document_type': document_type,
                    'document_face_confidence': round(doc_face_confidence, 2),
                    'live_face_confidence': round(live_face_confidence, 2),
                    'message': 'Identity verification failed. The face in the document does not match your live face.',
                    'next_step': 'Please ensure you are using your own ID document and try again.'
                })
                
        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'Could not compare faces. Please ensure both images have clear, visible faces.',
                'document_type': document_type
            }), 500
            
    except Exception as e:
        logger.error(f"Document verification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sessions/start', methods=['POST'])
def api_start_session():
    """REST API to start a new session"""
    data = request.get_json() or {}
    session_id = data.get('session_id') or str(uuid.uuid4())
    job_data = data.get('job_data', {})
    verification_token = data.get('verification_token')  # From pre-verification
    
    with session_lock:
        if session_id in active_sessions and active_sessions[session_id].is_active:
            return jsonify({
                'success': False, 
                'error': 'Session already active', 
                'session_id': session_id
            }), 400
        
        session = LiveInterviewSession(session_id, job_data)
        active_sessions[session_id] = session
    
    # Check if verification token was provided
    verification_applied = False
    verification_data = None
    
    if verification_token:
        with verification_lock:
            if verification_token in verified_documents:
                verification_data = verified_documents[verification_token]
                
                # Check if token has expired
                expires_at = datetime.fromisoformat(verification_data['expires_at'])
                if datetime.now() < expires_at:
                    # Apply verification to session
                    session.document_verified = verification_data['verified']
                    session.document_type = verification_data['document_type']
                    session.document_verification_result = {
                        'verified': verification_data['verified'],
                        'similarity': verification_data['similarity'],
                        'document_type': verification_data['document_type'],
                        'timestamp': verification_data['timestamp']
                    }
                    
                    # Auto-register face from verification
                    if verification_data.get('live_face_image'):
                        session.reference_face_image = verification_data['live_face_image']
                        session.face_registered = True
                    
                    verification_applied = True
                    logger.info(f"Applied verification token to session {session_id}")
                    
                    # Remove used token
                    del verified_documents[verification_token]
                else:
                    logger.warning(f"Verification token expired for session {session_id}")
    
    logger.info(f"Started session via REST: {session_id} (verification_applied: {verification_applied})")
    
    response_data = {
        'success': True,
        'session_id': session_id,
        'status': 'active',
        'start_time': session.start_time.isoformat(),
        'document_verified': session.document_verified,
        'face_registered': session.face_registered,
        'face_verification_required': not session.face_registered,
        'websocket_url': f'ws://{request.host}/socket.io/',
        'api_endpoints': {
            'analyze_frame': f'/api/sessions/{session_id}/frame',
            'get_metrics': f'/api/sessions/{session_id}/metrics',
            'get_verification_status': f'/api/sessions/{session_id}/verification-status',
            'register_face': f'/api/sessions/{session_id}/register-face',
            'end_session': f'/api/sessions/{session_id}/end'
        }
    }
    
    # Add verification info if token was used
    if verification_applied and verification_data:
        response_data['verification_info'] = {
            'document_type': verification_data['document_type'],
            'similarity': verification_data['similarity'],
            'verified_at': verification_data['timestamp']
        }
        response_data['message'] = 'Session started with verified identity. You can proceed directly to the interview.'
    else:
        response_data['message'] = 'Session started. Please verify your identity before proceeding.'
        response_data['pre_verification_endpoint'] = '/api/verify-document'
    
    return jsonify(response_data)


@app.route('/api/sessions/<session_id>/register-face', methods=['POST'])
def api_register_face(session_id):
    """REST API to register candidate's face for verification"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        return jsonify({'success': False, 'error': 'Session has ended'}), 400
    
    if session.face_registered:
        return jsonify({
            'success': False, 
            'error': 'Face already registered for this session',
            'face_registered': True
        }), 400
    
    # Get face image from request
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'face' not in request.files:
                return jsonify({'success': False, 'error': 'No face image provided'}), 400
            face_bytes = request.files['face'].read()
        else:
            data = request.get_json()
            if not data or 'face' not in data:
                return jsonify({'success': False, 'error': 'No face image data provided'}), 400
            
            face_data = data['face']
            if ',' in face_data:
                face_data = face_data.split(',')[1]
            face_bytes = base64.b64decode(face_data)
        
        result = session.register_face(face_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sessions/<session_id>/verify-document', methods=['POST'])
def api_verify_document(session_id):
    """
    REST API to verify candidate's identity using ID document.
    Compares face from document (Aadhaar, PAN, Passport, etc.) with live face.
    """
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        return jsonify({'success': False, 'error': 'Session has ended'}), 400
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Get document image
        if 'document' not in data:
            return jsonify({'success': False, 'error': 'No document image provided. Please upload your ID document.'}), 400
        
        # Get live face image
        if 'live_face' not in data:
            return jsonify({'success': False, 'error': 'No live face image provided. Please capture your face.'}), 400
        
        # Get document type (optional)
        document_type = data.get('document_type', 'other')
        valid_document_types = ['aadhaar', 'pan', 'passport', 'driving_license', 'voter_id', 'other']
        if document_type not in valid_document_types:
            document_type = 'other'
        
        # Decode document image
        document_data = data['document']
        if ',' in document_data:
            document_data = document_data.split(',')[1]
        document_bytes = base64.b64decode(document_data)
        
        # Decode live face image
        live_face_data = data['live_face']
        if ',' in live_face_data:
            live_face_data = live_face_data.split(',')[1]
        live_face_bytes = base64.b64decode(live_face_data)
        
        # Perform document verification
        result = session.verify_document(document_bytes, live_face_bytes, document_type)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Document verification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sessions/<session_id>/document-status', methods=['GET'])
def api_get_document_status(session_id):
    """REST API to get document verification status"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        **session.get_document_verification_status()
    })


@app.route('/api/sessions/<session_id>/verification-status', methods=['GET'])
def api_get_verification_status(session_id):
    """REST API to get face verification status"""
    if session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'face_registered': session.face_registered,
        'verification_enabled': session.face_verification_enabled,
        'verification_summary': session.get_face_verification_summary(),
        'recent_verifications': session.verification_results[-10:]  # Last 10 verification results
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
    """REST API to end a session and get final report in required UI format"""
    if session_id not in active_sessions:
        return jsonify({'status': 'error', 'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    
    if not session.is_active:
        return jsonify({'status': 'error', 'error': 'Session already ended'}), 400
    
    # Generate and return report directly (not wrapped)
    report = session.end_session()
    
    return jsonify(report)


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
//nst API_BASE = 'https://fexo.deepvox.ai';
const API_BASE = 'http://localhost:5001';


const response = await fetch(`${API_BASE}/api/sessions/start`, {
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

const analysis = await fetch(`${API_BASE}/api/sessions/${session_id}/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: frameData })
});
const result = await analysis.json();
console.log('Eye Contact:', result.result.metrics.eye_contact);

// 3. End Session & Get Report
const report = await fetch(`${API_BASE}/api/sessions/${session_id}/end`, {
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

