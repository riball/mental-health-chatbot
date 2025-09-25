import json
import random
import torch
import torch.nn as nn
import nltk
import gradio as gr
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# -------------------- Chatbot Model --------------------
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# -------------------- Chatbot Assistant --------------------
class ChatbotAssistant:
    def __init__(self, intents_path):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.X = None
        self.y = None

        self.override_map = {
            "im feeling sad": "stress",
            "i am feeling sad": "stress",
            "cheer me up": "gratitude",
            "make me happy": "gratitude",
            "i feel anxious": "anxiety",
            "im anxious": "anxiety"
        }

    @staticmethod
    def normalize_text(text):
        text = text.lower()
        text = text.replace("im", "i am")
        text = text.replace("can't", "cannot")
        text = text.replace("dont", "do not")
        text = text.replace("'", "")
        return text

    def tokenize_and_lemmatize(self, text):
        try:
            text = self.normalize_text(text)
            lemmatizer = nltk.WordNetLemmatizer()
            words = nltk.word_tokenize(text)
            words = [lemmatizer.lemmatize(word.lower()) for word in words]
            return words
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return text.split()

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        try:
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))
        except Exception as e:
            print(f"Error parsing intents: {e}")
            # Set default values if file reading fails
            self.intents = ["greeting"]
            self.intents_responses = {"greeting": ["Hello! How can I help you?"]}
            self.vocabulary = ["hello", "hi"]

    def load_model(self, model_path, metadata_path):
        try:
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print(f"Model files not found: {model_path}, {metadata_path}")
                return False
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.vocabulary = metadata['vocabulary']
            self.intents = metadata['intents']
            
            self.model = ChatbotModel(metadata['input_size'], metadata['output_size'])
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_message(self, input_message, confidence_threshold=0.6):
        try:
            if not input_message or not input_message.strip():
                return "Please enter a message."
                
            text = self.normalize_text(input_message)

            # Check override map
            if text in self.override_map:
                predicted_intent = self.override_map[text]
                if self.intents_responses.get(predicted_intent):
                    return random.choice(self.intents_responses[predicted_intent])

            # If model is not loaded, use simple pattern matching
            if self.model is None:
                return self.simple_response(text)

            words = self.tokenize_and_lemmatize(text)
            bag = self.bag_of_words(words)
            bag_tensor = torch.tensor([bag], dtype=torch.float32)

            with torch.no_grad():
                predictions = self.model(bag_tensor)
                probabilities = torch.softmax(predictions, dim=1)
                confidence, predicted_index = torch.max(probabilities, dim=1)

            if confidence.item() < confidence_threshold:
                return "I'm sorry, I did not understand that. Could you rephrase?"

            predicted_intent = self.intents[predicted_index]

            if self.intents_responses.get(predicted_intent):
                return random.choice(self.intents_responses[predicted_intent])
            else:
                return "I'm sorry, I did not understand that."
                
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I'm experiencing some technical difficulties. Please try again."

    def simple_response(self, text):
        """Fallback method when model is not available"""
        greetings = ["hello", "hi", "hey", "good morning", "good evening"]
        goodbyes = ["bye", "goodbye", "see you", "good night"]
        stress_keywords = ["stressed", "anxious", "overwhelmed", "panic", "worried"]
        
        if any(word in text for word in greetings):
            return "Hello! How are you feeling today?"
        elif any(word in text for word in goodbyes):
            return "Take care! Remember to be kind to yourself."
        elif any(word in text for word in stress_keywords):
            return "Take a deep breath. Focus on one thing at a time."
        else:
            return "I'm here to help. How are you feeling today?"

# -------------------- Initialize assistant --------------------
def initialize_assistant():
    try:
        assistant = ChatbotAssistant('intents.json')
        assistant.parse_intents()
        
        # Try to load the model, but continue even if it fails
        model_loaded = assistant.load_model('chatbot_model.pth', 'metadata.json')
        if not model_loaded:
            print("Warning: Could not load trained model. Using fallback responses.")
            
        return assistant
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        # Return a minimal assistant that can still respond
        assistant = ChatbotAssistant('intents.json')
        assistant.intents_responses = {
            "greeting": ["Hello! How can I help you today?"],
            "default": ["I'm here to help. How are you feeling?"]
        }
        return assistant

assistant = initialize_assistant()

# -------------------- Gradio Function --------------------
def chat_gradio(user_input):
    try:
        if not user_input or not user_input.strip():
            return "Please enter a message to chat with me."
        return assistant.process_message(user_input)
    except Exception as e:
        print(f"Error in chat_gradio: {e}")
        return "I'm sorry, I'm having trouble right now. Please try again."

# -------------------- Gradio Interface --------------------
def create_interface():
    iface = gr.Interface(
        fn=chat_gradio,
        inputs=gr.Textbox(
            lines=2, 
            placeholder="Type your message here...",
            label="Your Message"
        ),
        outputs=gr.Textbox(
            label="Chatbot Response",
            lines=3
        ),
        title="🤖 AI Mental Health Support Chatbot",
        description="A supportive AI chatbot to help with stress, anxiety, and mental wellness. Feel free to share how you're feeling.",
        theme=gr.themes.Soft(),
        examples=[
            ["Hello, how are you?"],
            ["I'm feeling stressed"],
            ["I need some motivation"],
            ["Can you give me self-care tips?"]
        ]
    )
    return iface

if __name__ == "__main__":
    try:
        iface = create_interface()
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"Error launching interface: {e}")