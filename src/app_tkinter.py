import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from rmn import RMN
import cv2
import threading
import time
from collections import defaultdict

# Italian translations of Plutchik's wheel emotions
PLUTCHIK_EMOTIONS = {
    'happy': ['Estasi', 'Gioia', 'Serenità', 'Ottimismo'],
    'sad': ['Dolore', 'Tristezza', 'Malinconia', 'Rimpianto'],
    'angry': ['Furia', 'Rabbia', 'Fastidio', 'Insofferenza'],
    'surprise': ['Stupore', 'Sorpresa', 'Perplessità', 'Incredulità'],
    'fear': ['Terrore', 'Paura', 'Apprensione', 'Ansia'],
    'disgust': ['Ripugnanza', 'Disgusto', 'Aversione', 'Disapprovazione'],
    'neutral': ['Calma', 'Neutralità', 'Indifferenza', 'Equilibrio']
}

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Magic Mirror")
        self.geometry("1000x1000")
        self.resizable(False, False)
        
        # Emotion detection setup
        self.emotion_detector = EmotionDetector()
        self.is_detecting = False
        self.face_detected_time = None
        self.emotion_confirmed = False
        self.selected_emotion = None
        self.emotion_counter = defaultdict(int)
        self.last_face_time = None
        
        # User responses storage
        self.user_responses = {
            'event': None,
            'location': None,
            'reaction': None
        }
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Center frame for all content
        self.center_frame = ctk.CTkFrame(self.main_frame)
        self.center_frame.pack(expand=True, fill="both")

        # Video display (always visible)
        #self.video_canvas = tk.Canvas(self.main_frame, width=640, height=480)
        #self.video_canvas.pack()

        # Status label (always visible)
        #self.status_label = ctk.CTkLabel(
        #    self.main_frame, 
        #    text="Avvicinati allo specchio...",
        #    font=("Arial", 16)
        #)
        #self.status_label.pack(pady=10)

        # Start with initial question
        self.show_initial_question()

        # Start detection automatically
        self.start_detection()

    def clear_center_frame(self):
        """Destroy all widgets in the center frame"""
        for widget in self.center_frame.winfo_children():
            widget.destroy()

    def show_initial_question(self):
        self.clear_center_frame()
        
        # Main question label
        self.question_label = ctk.CTkLabel(
            self.center_frame, 
            text="Come stai oggi?",
            font=("Arial", 28, "bold")
        )
        self.question_label.pack(expand=True)

    def start_detection(self):
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_thread = threading.Thread(target=self.update_detection, daemon=True)
            self.detection_thread.start()

    def update_detection(self):
        while self.is_detecting:
            frame, num_faces, emotion = self.emotion_detector.detect_emotion()
            
            if frame is not None:
                # Convert the frame to RGB and resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to ImageTk format
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update the canvas
                #self.video_canvas.create_image(0, 0, anchor="nw", image=imgtk)
                #self.video_canvas.image = imgtk

                current_time = time.time()
                
                # Track face detection and emotions
                if num_faces == 1:
                    self.last_face_time = current_time
                    if emotion:
                        self.emotion_counter[emotion] += 1
                
                # Check if we should proceed with emotion detection
                if not self.emotion_confirmed:
                    if self.face_detected_time is None and num_faces == 1:
                        self.face_detected_time = current_time
                        #self.status_label.configure(text="Rilevamento volto in corso...")
                    
                    # If we have face detection data for 5 seconds OR timeout with some detections
                    if ((self.face_detected_time and current_time - self.face_detected_time >= 5) or
                        (self.last_face_time and current_time - self.last_face_time > 2 and len(self.emotion_counter) > 0)):
                        
                        self.emotion_confirmed = True
                        # Get the most frequent emotion
                        if self.emotion_counter:
                            dominant_emotion = max(self.emotion_counter.items(), key=lambda x: x[1])[0]
                            self.show_emotion_buttons(dominant_emotion)
                        #else:
                            #self.status_label.configure(text="Nessuna emozione rilevata")
                
                # Reset if face is lost for too long
                elif num_faces != 1 and self.last_face_time and current_time - self.last_face_time > 5:
                    self.reset_detection()
            
            # Small delay to prevent freezing
            self.after(10, self.update_idletasks)

    def show_emotion_buttons(self, emotion):
        self.clear_center_frame()
        #self.status_label.configure(text="Emozione rilevata!")
        
        # Get base emotion and related emotions
        base_emotion = self.get_base_emotion(emotion)
        related_emotions = PLUTCHIK_EMOTIONS.get(base_emotion, [])
        
        # Create detected emotion label
        self.detected_emotion_label = ctk.CTkLabel(
            self.center_frame,
            text=f"Emozione principale: {emotion.capitalize()}",
            font=("Arial", 20, "bold")
        )
        self.detected_emotion_label.pack(pady=10)
        
        # Create related emotions buttons
        if related_emotions:
            related_label = ctk.CTkLabel(
                self.center_frame,
                text="Seleziona la tua emozione:",
                font=("Arial", 16)
            )
            related_label.pack()
            
            # Create buttons in a 2x2 grid
            for i, rel_emotion in enumerate(related_emotions):
                btn = ctk.CTkButton(
                    self.center_frame,
                    text=rel_emotion,
                    font=("Arial", 16),
                    width=180,
                    height=60,
                    command=lambda e=rel_emotion: self.start_question_flow(e)
                )
                btn.pack(pady=5)

    def start_question_flow(self, emotion):
        self.selected_emotion = emotion
        self.user_responses = {
            'emotion': emotion,
            'event': None,
            'location': None,
            'reaction': None
        }
        self.show_question(0)

    def show_question(self, question_index):
        self.clear_center_frame()
        
        questions = [
            "Qual è l'evento scatenante?",
            "Dove senti l'emozione?",
            "Quale reazione ti porta ad avere?"
        ]
        
        if question_index >= len(questions):
            self.show_final_summary()
            return
        
        # Current question label
        self.current_question_label = ctk.CTkLabel(
            self.center_frame,
            text=questions[question_index],
            font=("Arial", 20, "bold")
        )
        self.current_question_label.pack(pady=20)
        
        # Input field
        self.answer_input = ctk.CTkEntry(
            self.center_frame,
            width=400,
            height=40,
            font=("Arial", 16)
        )
        self.answer_input.pack(pady=20)
        self.answer_input.focus_set()
        
        # Bind Enter key to save response and show next question
        self.answer_input.bind('<Return>', 
            lambda event, idx=question_index: self.save_response_and_continue(idx))

    def save_response_and_continue(self, question_index):
        response = self.answer_input.get().strip()
        if response:
            # Save response based on question index
            if question_index == 0:
                self.user_responses['event'] = response
            elif question_index == 1:
                self.user_responses['location'] = response
            elif question_index == 2:
                self.user_responses['reaction'] = response
            
            # Clear input and show next question
            self.answer_input.delete(0, tk.END)
            self.show_question(question_index + 1)

    def show_final_summary(self):
        self.clear_center_frame()
        
        summary_text = (
            f"Evento: {self.user_responses['event']}\n\n"
            f"Emozione: {self.user_responses['emotion']}\n\n"
            f"Reazione: {self.user_responses['reaction']}"
        )
        
        self.summary_label = ctk.CTkLabel(
            self.center_frame,
            text=summary_text,
            font=("Arial", 20),
            justify="left"
        )
        self.summary_label.pack(expand=True)
        
        # Add restart button
        #self.restart_button = ctk.CTkButton(
        #    self.center_frame,
        #    text="Ricominciare",
        #    font=("Arial", 16),
        #    command=self.reset_detection
        #)
        #self.restart_button.pack(pady=20)

    def reset_detection(self):
        self.face_detected_time = None
        self.emotion_confirmed = False
        self.emotion_counter.clear()
        self.last_face_time = None
        self.selected_emotion = None
        self.user_responses = {
            'event': None,
            'location': None,
            'reaction': None
        }
        #self.status_label.configure(text="Avvicinati allo specchio...")
        self.show_initial_question()

    def get_base_emotion(self, emotion):
        emotion = emotion.lower()
        if 'happy' in emotion:
            return 'happy'
        elif 'sad' in emotion:
            return 'sad'
        elif 'angry' in emotion:
            return 'angry'
        elif 'surprise' in emotion:
            return 'surprise'
        elif 'fear' in emotion:
            return 'fear'
        elif 'disgust' in emotion:
            return 'disgust'
        return 'neutral'

    def on_closing(self):
        self.is_detecting = False
        self.emotion_detector.release()
        self.destroy()

class EmotionDetector:
    def __init__(self):
        self.model = RMN()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def detect_emotion(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading from webcam")
            return None, 0, None

        results = self.model.detect_emotion_for_single_frame(frame)
        num_faces = len(results) if results else 0
        
        # Get the dominant emotion
        emotion = None
        if num_faces == 1:
            try:
                emotion = results[0]['emo_label']
                if isinstance(emotion, dict):
                    emotion = max(emotion.items(), key=lambda x: x[1])[0]
            except (KeyError, TypeError, AttributeError):
                emotion = "neutral"
        
        # Draw the emotion detection on the frame
        frame = self.model.draw(frame, results)
        return frame, num_faces, emotion
    
    def release(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()