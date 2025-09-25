# Mental Health Chatbot Training in Google Colab
# Run each cell in order

# ================================
# CELL 1: Install Dependencies
# ================================
!pip install torch nltk numpy

# ================================
# CELL 2: Import Libraries and Download NLTK Data
# ================================
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import numpy as np
from collections import Counter

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Setup complete!")

# ================================
# CELL 3: Create intents.json file
# ================================
intents_data = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["hello", "hi", "hey", "good morning", "good evening", "yo", "howdy", "hiya", "hey there"],
      "responses": ["Hello! How are you feeling today?", "Hi there! How's your day going?", "Hey! I'm here to chat if you need me.", "Hello! Hope you're doing okay today."]
    },
    {
      "tag": "goodbye",
      "patterns": ["bye", "see you", "good night", "talk later", "catch you later", "see ya", "bye bye", "farewell"],
      "responses": ["Take care! Remember to be kind to yourself.", "Goodbye! I hope you have a calm day.", "See you later! Stay positive.", "Bye! Reach out if you need to talk again."]
    },
    {
      "tag": "stress",
      "patterns": ["I am stressed", "feeling anxious", "I can't handle this", "overwhelmed", "pressure", "so much to do", "panic", "nervous", "worried"],
      "responses": ["Take a deep breath. Focus on one thing at a time.", "Stress is temporary. You're doing your best, and that's enough.", "Try a short walk or some gentle stretching to calm down.", "Close your eyes, take slow breaths, and remind yourself you're safe.", "Talking to a friend or journaling can help relieve pressure."]
    },
    {
      "tag": "motivation",
      "patterns": ["I feel hopeless", "I can't do this", "I give up", "I need motivation", "nothing works", "so tired of trying", "I feel stuck"],
      "responses": ["Every step counts. Keep going, you're stronger than you think.", "Take it one moment at a time. You've got this!", "Progress is progress, no matter how small.", "Remember, even small victories are wins.", "You have overcome challenges before; you can do it again!"]
    },
    {
      "tag": "self_care",
      "patterns": ["self care tips", "how can I relax", "I want to feel better", "help me calm down", "me time", "need to unwind", "take care of myself"],
      "responses": ["Drink water, stretch, or take a short walk.", "Listening to music, journaling, or meditating can calm your mind.", "Try some deep breathing or light exercise.", "Spend time doing something you enjoy for a few minutes.", "Remember: rest is productive, too!"]
    },
    {
      "tag": "resources",
      "patterns": ["I need help", "I feel suicidal", "I need support", "hotline", "help me", "I want to talk", "I feel unsafe", "I feel hopeless"],
      "responses": ["If you are in crisis, please contact a trained professional immediately.", "You can reach out to the Vandrevala Foundation Helpline: 1860 266 2345 (India).", "You can also contact iCall: +91 9152987821 (India) for support.", "If you feel unsafe, please reach out to a local emergency number immediately.", "It's okay to ask for help. You are not alone."]
    },
    {
      "tag": "loneliness",
      "patterns": ["I feel lonely", "no one understands me", "I have no friends", "feeling isolated", "alone"],
      "responses": ["It's okay to feel lonely sometimes. Talking helps.", "Even reaching out online or to a helpline can make a difference.", "You are not alone in feeling this way. There are people who care.", "Connecting with communities or groups can help you feel supported."]
    },
    {
      "tag": "anxiety",
      "patterns": ["I feel anxious", "panic attacks", "worried all the time", "my heart races", "I can't calm down"],
      "responses": ["Take slow, deep breaths and ground yourself in the present.", "Focus on something around you — notice 5 things you can see, 4 things you can touch.", "Sometimes writing down your worries can help you manage them.", "Gentle exercise or stretching may help reduce anxious feelings."]
    },
    {
      "tag": "sleep",
      "patterns": ["I can't sleep", "insomnia", "trouble sleeping", "nightmares", "I feel tired but can't rest"],
      "responses": ["Try a consistent bedtime and reduce screen time before sleep.", "Relax with calming music, reading, or meditation.", "Avoid caffeine and heavy meals close to bedtime.", "Practice deep breathing to help calm your mind."]
    },
    {
      "tag": "gratitude",
      "patterns": ["I want to feel better", "cheer me up", "make me happy", "I need positivity", "uplift me"],
      "responses": ["Take a moment to appreciate one small good thing today.", "Write down 3 things you're thankful for today.", "Remember, small moments of joy matter as much as big ones.", "Even a short walk outside or talking to someone you care about can lift your mood."]
    },
    {
      "tag": "thanks",
      "patterns": ["thanks", "thank you", "I appreciate it", "good bot", "thanks a lot"],
      "responses": ["You're welcome!", "Glad I could help!", "Anytime! Take care of yourself.", "Always here if you need me."]
    }
  ]
}

# Save intents.json
with open('intents.json', 'w') as f:
    json.dump(intents_data, f, indent=2)

print("intents.json created successfully!")

# ================================
# CELL 4: Define Model Architecture
# ================================
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

print("Model architecture defined!")

# ================================
# CELL 5: Training Class
# ================================
class ChatbotTrainer:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.X = []
        self.y = []

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
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
            return words
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return text.split()

    def parse_intents(self):
        """Parse the intents.json file and prepare training data"""
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        # Collect all words and intents
        all_words = []
        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                all_words.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))

        # Create vocabulary (remove duplicates and sort)
        self.vocabulary = sorted(set(all_words))
        print(f"📊 Vocabulary size: {len(self.vocabulary)}")
        print(f"📊 Number of intents: {len(self.intents)}")
        print(f"📊 Number of patterns: {len(self.documents)}")

    def create_training_data(self):
        """Create bag of words training data"""
        for doc_words, intent_tag in self.documents:
            # Create bag of words
            bag = [1 if word in doc_words else 0 for word in self.vocabulary]
            self.X.append(bag)
            
            # Create output label (one-hot encoded)
            output = [0] * len(self.intents)
            output[self.intents.index(intent_tag)] = 1
            self.y.append(output)

        # Convert to numpy arrays
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        print(f"✅ Training data shape: {self.X.shape}")
        print(f"✅ Training labels shape: {self.y.shape}")

    def train_model(self, epochs=1000, learning_rate=0.001):
        """Train the neural network model"""
        input_size = len(self.vocabulary)
        output_size = len(self.intents)
        
        # Initialize model
        model = ChatbotModel(input_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(self.X)
        y_tensor = torch.tensor(self.y)

        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        print("🎉 Training completed!")
        return model

    def save_model_and_metadata(self, model, model_path='chatbot_model.pth', metadata_path='metadata.json'):
        """Save the trained model and metadata"""
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'input_size': len(self.vocabulary),
            'output_size': len(self.intents)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {model_path}")
        print(f"💾 Metadata saved to {metadata_path}")

    def test_model(self, model, test_phrases):
        """Test the trained model with some example phrases"""
        model.eval()
        print("\n🧪 Testing Model:")
        print("=" * 50)
        
        with torch.no_grad():
            for phrase in test_phrases:
                words = self.tokenize_and_lemmatize(phrase)
                bag = [1 if word in words else 0 for word in self.vocabulary]
                bag_tensor = torch.tensor([bag], dtype=torch.float32)
                
                predictions = model(bag_tensor)
                probabilities = torch.softmax(predictions, dim=1)
                confidence, predicted_index = torch.max(probabilities, dim=1)
                
                predicted_intent = self.intents[predicted_index.item()]
                
                print(f"📝 Input: '{phrase}'")
                print(f"🎯 Predicted Intent: {predicted_intent}")
                print(f"📊 Confidence: {confidence.item():.3f}")
                print("-" * 30)

print("ChatbotTrainer class defined!")

# ================================
# CELL 6: Run Training
# ================================
print("🤖 Starting Chatbot Training Process...")
print("=" * 60)

# Initialize trainer
trainer = ChatbotTrainer('intents.json')

# Parse intents and prepare data
print("📚 Parsing intents...")
trainer.parse_intents()

print("\n🔢 Creating training data...")
trainer.create_training_data()

print("\n🏋️ Training model...")
model = trainer.train_model(epochs=1000, learning_rate=0.001)

print("\n💾 Saving model and metadata...")
trainer.save_model_and_metadata(model)

# Test the model
test_phrases = [
    "hello there",
    "i am really stressed out",
    "feeling very anxious today", 
    "thank you so much",
    "goodbye for now",
    "i really need some motivation",
    "give me self care tips",
    "i feel so lonely",
    "can't sleep at night"
]

trainer.test_model(model, test_phrases)

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("📁 Files created:")
print("   ✅ chatbot_model.pth")
print("   ✅ metadata.json") 
print("   ✅ intents.json")
print("\n🚀 Ready for deployment to Hugging Face!")

# ================================
# CELL 7: Download Files (Optional)
# ================================
from google.colab import files

print("📥 Downloading trained files...")

# Download the model file
files.download('chatbot_model.pth')

# Download the metadata file  
files.download('metadata.json')

# Download intents file (if you want to verify)
files.download('intents.json')

print("✅ All files downloaded! Upload them to your Hugging Face Space.")

# ================================
# CELL 8: Quick Test Function (Optional)
# ================================
def quick_test_model():
    """Quick function to test the model interactively"""
    print("🤖 Quick Model Test - Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        words = trainer.tokenize_and_lemmatize(user_input)
        bag = [1 if word in words else 0 for word in trainer.vocabulary]
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            predictions = model(bag_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
            
            predicted_intent = trainer.intents[predicted_index.item()]
            
            print(f"Bot: Intent detected: {predicted_intent} (confidence: {confidence.item():.3f})")
            
            # Get a random response from the intent
            if predicted_intent in ['greeting', 'goodbye', 'stress', 'motivation', 'self_care', 'anxiety', 'gratitude']:
                responses = {
                    'greeting': ["Hello! How are you feeling today?", "Hi there! How's your day going?"],
                    'goodbye': ["Take care! Remember to be kind to yourself.", "Goodbye! I hope you have a calm day."],
                    'stress': ["Take a deep breath. Focus on one thing at a time.", "Stress is temporary. You're doing your best."],
                    'motivation': ["Every step counts. Keep going, you're stronger than you think!", "Take it one moment at a time."],
                    'self_care': ["Drink water, stretch, or take a short walk.", "Try some deep breathing or light exercise."],
                    'anxiety': ["Take slow, deep breaths and ground yourself in the present.", "Focus on something around you."],
                    'gratitude': ["Take a moment to appreciate one small good thing today.", "Write down 3 things you're thankful for."]
                }
                response = random.choice(responses.get(predicted_intent, ["I'm here to help!"]))
                print(f"Bot: {response}")
            print("-" * 40)

# Uncomment the line below if you want to test interactively
# quick_test_model()

print("🎯 Training notebook complete! Your chatbot is ready!")