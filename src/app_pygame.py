import os
import time
import threading
import json
import cv2
import numpy as np
import pygame
import whisper
import pyaudio
import wave
import tempfile
from rmn import RMN  # Import the Residual Masking Network

# Configuration
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
LANGUAGE = 'it'  # Italian
RESET_TIMEOUT = 60  # seconds before resetting to initial state
FACE_DETECTION_DURATION = 5  # seconds required for consistent face detection
SILENCE_THRESHOLD = 0.02  # Threshold for silence detection
SILENCE_TIMEOUT = 5  # seconds of silence before auto-submitting answer
SPEAKING_TIMEOUT = 1.5  # seconds of audio gathering before processing with Whisper

# Plutchik's wheel of emotions - mapping from base emotions to related emotions
EMOTION_MAPPING = {
    "happy": ["estasi", "serenità", "amore", "ottimismo"],
    "sad": ["malinconia", "delusione", "solitudine", "dispiacere"],
    "angry": ["rabbia", "irritazione", "fastidio", "indignazione"],
    "fear": ["terrore", "ansia", "preoccupazione", "inquietudine"],
    "surprise": ["stupore", "meraviglia", "incredulità", "confusione"],
    "disgust": ["repulsione", "avversione", "disprezzo", "disapprovazione"],
    "neutral": ["calma", "tranquillità", "indifferenza", "distacco"]
}

# Mapping from RMN emotion labels to our base categories
RMN_TO_BASE_EMOTION = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    # Add additional mappings if RMN has different labels
}

# Emoji mapping for emotions
EMOTION_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

class WhisperVoiceRecognition(threading.Thread):
    """Thread class for voice recognition using OpenAI's Whisper"""
    def __init__(self, language='it'):
        threading.Thread.__init__(self)
        self.language = language
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model("small")  # Use "tiny", "base", "small", "medium", or "large"
        print("Whisper model loaded.")
        
        # Audio setup
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        
        self.is_listening = False
        self.stop_listening = False
        self.transcript = ""
        self.temp_transcript = ""
        self.last_voice_time = time.time()
        self.is_speaking = False
        self.audio_frames = []
        self.daemon = True  # Thread will close when main program exits
        
        # For emotion selection tracking
        self.detected_words = set()
        
    def run(self):
        """Main method for the thread"""
        try:
            while not self.stop_listening:
                if self.is_listening:
                    # If not actively recording audio for processing
                    if not self.is_speaking:
                        # Start stream for volume level detection
                        stream = self.audio.open(format=self.format,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input=True,
                                    frames_per_buffer=self.chunk)
                        
                        # Check volume level to detect speaking
                        try:
                            data = np.frombuffer(stream.read(self.chunk, exception_on_overflow=False), dtype=np.int16)
                            volume_norm = np.abs(data).mean() / 32768.0
                            
                            if volume_norm > SILENCE_THRESHOLD:
                                # Speaking detected
                                print(f"Speaking detected: {volume_norm}")
                                self.last_voice_time = time.time()
                                
                                # Start collecting audio for processing
                                self.is_speaking = True
                                self.audio_frames = [data]
                                record_start_time = time.time()
                                
                                # Collect audio for SPEAKING_TIMEOUT seconds
                                while time.time() - record_start_time < SPEAKING_TIMEOUT:
                                    data = np.frombuffer(stream.read(self.chunk, exception_on_overflow=False), dtype=np.int16)
                                    self.audio_frames.append(data)
                                    volume_norm = np.abs(data).mean() / 32768.0
                                    if volume_norm > SILENCE_THRESHOLD:
                                        # Reset timer if still speaking
                                        record_start_time = time.time()
                                
                                # Close the stream before processing with Whisper
                                stream.stop_stream()
                                stream.close()
                                
                                # Process audio with Whisper
                                self.process_audio_with_whisper()
                                
                                # Reset for next detection
                                self.is_speaking = False
                            
                        except Exception as e:
                            print(f"Error in audio processing: {e}")
                            if stream:
                                stream.stop_stream()
                                stream.close()
                            self.is_speaking = False
                        
                        if stream:
                            stream.stop_stream()
                            stream.close()
                
                time.sleep(0.1)  # Sleep to prevent high CPU usage
                
        except Exception as e:
            print(f"Voice recognition thread error: {e}")
        
    def process_audio_with_whisper(self):
        """Process the recorded audio with Whisper"""
        try:
            # Save the audio frames to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
            
            # Process with Whisper
            result = self.model.transcribe(
                temp_filename, 
                language=self.language,
                fp16=False  # Set to True if you have GPU support
            )
            
            # Update transcript
            if result["text"].strip():
                self.temp_transcript = result["text"].strip()
                self.transcript += " " + self.temp_transcript if self.transcript else self.temp_transcript
                print(f"Whisper recognized: {self.temp_transcript}")
                
                # Update detected words for emotion selection
                for word in self.temp_transcript.lower().split():
                    self.detected_words.add(word)
                
                self.last_voice_time = time.time()
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"Error processing audio with Whisper: {e}")

    def check_emotion_mentioned(self, emotion_words):
        """Versione migliorata per il controllo delle emozioni menzionate"""
        transcript_lower = self.transcript.lower()
        detected_words_lower = {word.lower() for word in self.detected_words}
        
        # 1. Controllo diretto delle parole
        direct_matches = [word for word in emotion_words 
                         if word.lower() in transcript_lower or 
                         word.lower() in detected_words_lower]
        
        if direct_matches:
            return direct_matches[0]  # Restituisce la prima corrispondenza diretta
        
        # 2. Controllo di similarità (per gestire errori di riconoscimento)
        from difflib import get_close_matches
        
        for emotion in emotion_words:
            # Cerca corrispondenze approssimative nel transcript
            matches = get_close_matches(emotion.lower(), transcript_lower.split(), 
                                      n=1, cutoff=0.7)
            if matches:
                return emotion
                
        # 3. Controllo di parole chiave correlate
        emotion_keywords = {
            "estasi": ["felice", "gioia", "contento"],
            "serenità": ["calma", "pacifico", "tranquillo"],
            "amore": ["amore", "affetto", "passione"],
            "ottimismo": ["speranza", "fiducia", "positivo"],
            # Aggiungi altre mappature...
        }
        
        for emotion in emotion_words:
            keywords = emotion_keywords.get(emotion.lower(), [])
            for keyword in keywords:
                if keyword in transcript_lower:
                    return emotion
                    
        return None
    
    def start_listening(self):
        """Start listening for speech"""
        self.is_listening = True
        self.last_voice_time = time.time()
        
    def stop_listening_now(self):
        """Stop listening for speech"""
        self.is_listening = False
        
    def reset_transcript(self):
        """Reset the transcript"""
        self.transcript = ""
        self.temp_transcript = ""
        self.detected_words = set()
        
    def get_transcript(self):
        """Get the current transcript"""
        return self.transcript
        
    def get_silence_duration(self):
        """Get how long it's been silent"""
        return time.time() - self.last_voice_time
    
    def check_emotion_mentioned(self, emotion_words):
        """Check if any of the emotion words were detected in speech"""
        emotion_words_lower = [word.lower() for word in emotion_words]
        return any(word in self.detected_words for word in emotion_words_lower)
        
    def shutdown(self):
        """Shut down the voice recognition thread"""
        self.stop_listening = True
        self.is_listening = False

class EmotionMirror:
    def __init__(self):
        # Initialize pygame for display
        pygame.init()
        pygame.display.set_caption("Emotion Mirror")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font_large = pygame.font.SysFont('Arial', 46)
        self.font_medium = pygame.font.SysFont('Arial', 36)
        self.font_small = pygame.font.SysFont('Arial', 26)
        self.emotion_buttons = []
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize RMN for emotion recognition
        self.emotion_detector = RMN()
        
        # Initialize voice recognition with Whisper
        self.voice_recognition = WhisperVoiceRecognition(language=LANGUAGE)
        self.voice_recognition.start()
        
        # State variables
        self.state = "initial"  # States: initial, face_detection, detecting, selecting, questioning, summary
        self.detected_emotion = None
        self.selected_emotion = None
        self.question_index = 0
        self.question_answers = ["", "", ""]
        self.questions = [
            "Quale l'evento scatenante?",
            "Dove senti l'emozione nel tuo corpo?",
            "Quale reazione ti porta ad avere?"
        ]
        
        # Face detection timer variables
        self.face_detection_start_time = None
        self.continuous_face_detection = False
        self.face_detection_progress = 0.0  # 0.0 to 1.0 for progress bar
        
        # For background processes
        self.last_interaction_time = time.time()
    
    def detect_face(self, frame):
        """Detect if there's a face in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0, faces
    
    def detect_emotion(self, frame):
        """
        Use RMN to detect emotion in the frame
        """
        # Process the frame with RMN
        results = self.emotion_detector.detect_emotion_for_single_frame(frame)
        
        # RMN returns a list of detected faces with emotions
        if results and len(results) > 0:
            # Get the emotion of the first detected face
            emotion = results[0]['emo_label']
            
            # Map RMN emotion to our base emotion categories
            base_emotion = RMN_TO_BASE_EMOTION.get(emotion, "neutral")
            return base_emotion
        else:
            # Default to neutral if no face emotions are detected
            return "neutral"
    
    def render_initial_screen(self):
        """Render the initial question screen"""
        self.screen.fill((0, 0, 0))
        text = self.font_large.render("Quale espressione ti identifica oggi?", True, (255, 255, 255))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(text, text_rect)
        
        # Instructions for user
        instruction = self.font_medium.render("Posizionati davanti alla telecamera", True, (200, 200, 200))
        instruction_rect = instruction.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 80))
        self.screen.blit(instruction, instruction_rect)
        
        # Display camera feed in a small corner
        self.display_camera_feed(0, 0, 320, 240)
    
    def render_face_detection_screen(self):
        """Render the face detection waiting screen"""
        self.screen.fill((0, 0, 0))
        
        # Main text
        text = self.font_large.render("Rilevamento viso in corso...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//3))
        self.screen.blit(text, text_rect)
        
        # Progress bar for the 5-second detection
        bar_width = 600
        bar_height = 30
        bar_x = (WINDOW_WIDTH - bar_width) // 2
        bar_y = WINDOW_HEIGHT // 2
        
        # Background bar
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        # Progress fill
        fill_width = int(bar_width * self.face_detection_progress)
        pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, fill_width, bar_height))
        
        # Time remaining
        time_left = max(0, FACE_DETECTION_DURATION - self.face_detection_progress * FACE_DETECTION_DURATION)
        time_text = self.font_medium.render(f"Rimangono {time_left:.1f} secondi", True, (255, 255, 255))
        time_rect = time_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
        self.screen.blit(time_text, time_rect)
        
        # Display camera feed in a larger size for better feedback
        self.display_camera_feed((WINDOW_WIDTH - 480) // 2, WINDOW_HEIGHT//2 + 100, 480, 360)
    
    def render_emotion_selection(self):
        """Render the emotion selection screen with voice commands"""
        self.screen.fill((0, 0, 0))
        
        # Show detected emotion with emoji
        emoji = EMOTION_EMOJI.get(self.detected_emotion, "😐")
        detected_text = self.font_medium.render(f"Emozione rilevata: {self.detected_emotion} {emoji}", True, (255, 255, 255))
        self.screen.blit(detected_text, (100, 100))
        
        # Show related emotions for selection via voice
        selection_text = self.font_large.render("Dimmi quale emozione ti rappresenta:", True, (255, 255, 255))
        selection_rect = selection_text.get_rect(center=(WINDOW_WIDTH//2, 180))
        self.screen.blit(selection_text, selection_rect)
        
        # If voice recognition is active, start listening
        #if not self.voice_recognition.is_listening:
        #    self.voice_recognition.reset_transcript()
        #    self.voice_recognition.start_listening()
        
        # Show the spoken text
        transcript = self.voice_recognition.get_transcript()
        if transcript:
            # Display the recognized text
            transcript_surface = pygame.Surface((WINDOW_WIDTH - 200, 100))
            transcript_surface.fill((50, 50, 50))
            
            # Split into lines if needed
            words = transcript.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                if self.font_medium.size(test_line)[0] > WINDOW_WIDTH - 220:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Display the last few lines
            for i, line in enumerate(lines[-2:]):
                line_text = self.font_medium.render(line, True, (255, 255, 255))
                transcript_surface.blit(line_text, (10, 10 + i * 40))
            
            self.screen.blit(transcript_surface, (100, 230))
        
        # Show related emotions as options
        self.emotion_buttons = []
        related_emotions = EMOTION_MAPPING.get(self.detected_emotion, [])
        for i, emotion in enumerate(related_emotions):
            y_pos = 350 + i * 60
            emotion_rect = pygame.Rect(100, y_pos, WINDOW_WIDTH - 200, 50)
            pygame.draw.rect(self.screen, (50, 50, 50), emotion_rect)
            emotion_text = self.font_medium.render(emotion, True, (255, 255, 255))
            text_rect = emotion_text.get_rect(center=emotion_rect.center)
            self.screen.blit(emotion_text, text_rect)
            self.emotion_buttons.append((emotion, emotion_rect))
        
       # related_emotions = EMOTION_MAPPING.get(self.detected_emotion, [])
       # for i, emotion in enumerate(related_emotions):
       #     y_pos = 350 + i * 60
       #     emotion_rect = pygame.Rect(100, y_pos, WINDOW_WIDTH - 200, 50)
       #     
       #     # Highlight the emotion if it's mentioned in speech
       #     if self.voice_recognition.check_emotion_mentioned([emotion]):
       #         pygame.draw.rect(self.screen, (0, 150, 0), emotion_rect)
       #         
       #         # Auto-select after a short delay
       #         if self.voice_recognition.get_silence_duration() > 1.0:
       #             self.selected_emotion = emotion
       #             self.state = "questioning"
       #             self.question_index = 0
       #             self.voice_recognition.reset_transcript()
       #             break
       #     else:
       #         pygame.draw.rect(self.screen, (50, 50, 50), emotion_rect)
       #     
       #     emotion_text = self.font_medium.render(emotion, True, (255, 255, 255))
       #     text_rect = emotion_text.get_rect(center=emotion_rect.center)
       #     self.screen.blit(emotion_text, text_rect)
        
        # Instructions
        instruction_text = self.font_small.render("Pronuncia una delle emozioni elencate...", True, (200, 200, 200))
        instruction_rect = instruction_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT - 50))
        self.screen.blit(instruction_text, instruction_rect)
        
        # Display camera feed in a small corner
        self.display_camera_feed(0, 0, 320, 240)
    
    def render_question_screen(self):
        """Render the question screen with voice input"""
        self.screen.fill((0, 0, 0))
        
        # Show current question
        question = self.questions[self.question_index]
        question_text = self.font_large.render(question, True, (255, 255, 255))
        question_rect = question_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//3))
        self.screen.blit(question_text, question_rect)
        
        # If voice recognition is not active, start it
        if not self.voice_recognition.is_listening:
            self.voice_recognition.reset_transcript()
            self.voice_recognition.start_listening()
        
        # Display recognized speech
        transcript = self.voice_recognition.get_transcript()
        
        # Text area for speech recognition results
        input_rect = pygame.Rect(100, WINDOW_HEIGHT//2 - 50, WINDOW_WIDTH - 200, 100)
        pygame.draw.rect(self.screen, (50, 50, 50), input_rect)
        
        # Format and display the recognized text
        if transcript:
            words = transcript.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                # Check if current line is too long
                test_line = ' '.join(current_line)
                if self.font_medium.size(test_line)[0] > input_rect.width - 20:
                    # Remove the last word and add the line
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            # Add the last line
            if current_line:
                lines.append(' '.join(current_line))
            
            # Display the lines
            for i, line in enumerate(lines[-3:]):  # Show only the last 3 lines to fit in the box
                line_text = self.font_medium.render(line, True, (255, 255, 255))
                self.screen.blit(line_text, (input_rect.x + 10, input_rect.y + 10 + (i * 30)))
        
        # Instructions and status
        silence_duration = self.voice_recognition.get_silence_duration()
        if silence_duration > 2:  # Only show countdown after 2 seconds of silence
            time_left = max(0, SILENCE_TIMEOUT - silence_duration)
            instructions = self.font_small.render(f"Silenzio rilevato, invio tra {time_left:.1f} secondi...", True, (255, 100, 100))
        else:
            instructions = self.font_small.render("Parla ora... (il sistema registra automaticamente)", True, (100, 255, 100))
        
        instructions_rect = instructions.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 80))
        self.screen.blit(instructions, instructions_rect)
        
        # Progress indicator for silence timeout
        bar_width = 400
        bar_height = 10
        bar_x = (WINDOW_WIDTH - bar_width) // 2
        bar_y = WINDOW_HEIGHT//2 + 120
        
        # Background bar
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        # Progress fill (represents time until submission)
        fill_percentage = min(1.0, silence_duration / SILENCE_TIMEOUT)
        fill_width = int(bar_width * fill_percentage)
        pygame.draw.rect(self.screen, (255, 50, 50), (bar_x, bar_y, fill_width, bar_height))
        
        # Check if we should auto-submit due to silence
        if silence_duration > SILENCE_TIMEOUT and transcript:
            self.submit_answer()
    
    def render_summary_screen(self):
        """Render the summary timeline screen"""
        self.screen.fill((0, 0, 0))
        
        # Title
        title_text = self.font_large.render("Riepilogo dell'emozione", True, (255, 255, 255))
        self.screen.blit(title_text, (100, 80))
        
        # Timeline visualization
        pygame.draw.line(self.screen, (255, 255, 255), (150, 200), (WINDOW_WIDTH - 150, 200), 3)
        
        # Event
        event_text = self.font_medium.render("Evento scatenante:", True, (255, 255, 255))
        self.screen.blit(event_text, (150, 250))
        
        # Split answers into multi-line text if needed
        self.render_multiline_text(self.question_answers[0], 150, 290, WINDOW_WIDTH - 300)
        
        # Emotion with emoji
        emoji = EMOTION_EMOJI.get(self.detected_emotion, "😐")
        emotion_text = self.font_medium.render(f"Emozione: {self.selected_emotion} {emoji}", True, (255, 255, 255))
        self.screen.blit(emotion_text, (150, 360))
        
        # Body location
        body_text = self.font_medium.render("Dove nel corpo:", True, (255, 255, 255))
        self.screen.blit(body_text, (150, 410))
        
        self.render_multiline_text(self.question_answers[1], 150, 450, WINDOW_WIDTH - 300)
        
        # Reaction
        reaction_text = self.font_medium.render("Reazione:", True, (255, 255, 255))
        self.screen.blit(reaction_text, (150, 520))
        
        self.render_multiline_text(self.question_answers[2], 150, 560, WINDOW_WIDTH - 300)
        
        # Automatic reset after delay message
        time_left = max(0, RESET_TIMEOUT - (time.time() - self.last_interaction_time))
        reset_text = self.font_small.render(f"Nuova sessione tra {time_left:.0f} secondi...", True, (200, 200, 200))
        reset_rect = reset_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT - 50))
        self.screen.blit(reset_text, reset_rect)
    
    def render_multiline_text(self, text, x, y, max_width):
        """Render a multi-line text with word wrapping"""
        if not text:
            return
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            # Check if current line is too long
            test_line = ' '.join(current_line)
            if self.font_small.size(test_line)[0] > max_width:
                # Remove the last word and add the line
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # Display the lines (limit to 2 lines per answer)
        for i, line in enumerate(lines[:2]):
            line_text = self.font_small.render(line, True, (200, 200, 200))
            self.screen.blit(line_text, (x, y + (i * 25)))
            
        # Add ellipsis if there are more lines
        if len(lines) > 2:
            self.screen.blit(self.font_small.render("...", True, (200, 200, 200)), (x, y + 50))
    
    def display_camera_feed(self, x, y, width, height):
        """Display camera feed in a corner of the screen"""
        ret, frame = self.camera.read()
        if ret:
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw rectangles around detected faces
            has_face, faces = self.detect_face(frame)
            if has_face:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x_face, y_face, w_face, h_face) in faces:
                    cv2.rectangle(frame, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 0), 2)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            frame = cv2.resize(frame, (width, height))
            
            # Convert to pygame surface
            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (x, y))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.last_interaction_time = time.time()
                if self.state == "selecting":
                    mouse_pos = event.pos
                    for emotion, rect in self.emotion_buttons:
                        if rect.collidepoint(mouse_pos):
                            self.selected_emotion = emotion
                            self.state = "questioning"
                            self.question_index = 0
                            return True
            
            if event.type == pygame.KEYDOWN:
                self.last_interaction_time = time.time()

        return True
    
    def submit_answer(self):
        """Submit the current answer and move to next question or summary"""
        # Get the transcript
        transcript = self.voice_recognition.get_transcript().strip()
        if transcript:
            self.question_answers[self.question_index] = transcript
            
            self.question_index += 1
            if self.question_index >= len(self.questions):
                self.state = "summary"
                # Stop listening when reaching summary
                self.voice_recognition.stop_listening_now()
            else:
                # Reset for next question
                self.voice_recognition.reset_transcript()
    
    def check_for_face(self):
        """Check if there's a face in front of the camera"""
        ret, frame = self.camera.read()
        if not ret:
            return
        
        has_face, _ = self.detect_face(frame)
        current_time = time.time()
        
        # State transitions based on face detection
        if self.state == "initial" and has_face:
            # Start the 5-second face detection timer
            if not self.continuous_face_detection:
                self.face_detection_start_time = current_time
                self.continuous_face_detection = True
                self.state = "face_detection"
                print("Face detected, starting 5-second timer")
        
        elif self.state == "face_detection":
            if has_face:
                # Calculate progress (0.0 to 1.0)
                elapsed = current_time - self.face_detection_start_time
                self.face_detection_progress = min(1.0, elapsed / FACE_DETECTION_DURATION)
                
                # If we've reached the required duration with continuous face detection
                if self.face_detection_progress >= 1.0:
                    # Use RMN to detect emotion
                    self.detected_emotion = self.detect_emotion(frame)
                    self.state = "selecting"
                    self.last_interaction_time = current_time
                    print(f"Face detected for 5 seconds, emotion detected: {self.detected_emotion}")
            else:
                # Reset the timer if face detection is lost
                self.continuous_face_detection = False
                self.face_detection_progress = 0.0
                self.state = "initial"
                print("Face detection lost, resetting to initial state")
        
        # Check for timeout to reset
        if time.time() - self.last_interaction_time > RESET_TIMEOUT and self.state not in ["initial", "face_detection"]:
            self.reset_system()
    
    def reset_system(self):
        """Reset the system to initial state"""
        self.state = "initial"
        self.detected_emotion = None
        self.selected_emotion = None
        self.question_index = 0
        self.question_answers = ["", "", ""]
        self.voice_recognition.stop_listening_now()
        self.voice_recognition.reset_transcript()
        self.continuous_face_detection = False
        self.face_detection_progress = 0.0
        print("System reset to initial state")
    
    def run(self):
        """Main application loop"""
        running = True
        
        try:
            while running:
                # Handle pygame events
                running = self.handle_events()
                
                # Check for face if in initial or face_detection state
                if self.state in ["initial", "face_detection"]:
                    self.check_for_face()
                
                # Render the appropriate screen based on state
                if self.state == "initial":
                    self.render_initial_screen()
                elif self.state == "face_detection":
                    self.render_face_detection_screen()
                elif self.state == "selecting":
                    self.render_emotion_selection()
                elif self.state == "questioning":
                    self.render_question_screen()
                elif self.state == "summary":
                    self.render_summary_screen()
                    
                # Check for timeout to reset to initial state
                if time.time() - self.last_interaction_time > RESET_TIMEOUT:
                    self.reset_system()
                
                # Update the display
                pygame.display.flip()
                
                # Cap the frame rate
                pygame.time.delay(30)  # ~33 fps
        
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            # Clean up resources
            self.voice_recognition.shutdown()
            self.camera.release()
            pygame.quit()
            print("Application resources released")

# Add code to run the application if this is the main module
if __name__ == "__main__":
    try:
        emotion_mirror = EmotionMirror()
        emotion_mirror.run()
    except Exception as e:
        print(f"Application error: {e}")
        # Ensure pygame is properly quit
        pygame.quit()
