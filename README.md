🤖 AI Mental Health Support Chatbot
A supportive AI chatbot built with PyTorch and Gradio to help with stress, anxiety, and mental wellness conversations. The chatbot uses natural language processing to understand user inputs and provide appropriate supportive responses.
🌟 Features

Mental Health Support: Provides responses for stress, anxiety, depression, and general wellness
Natural Language Processing: Uses NLTK for text processing and PyTorch for intent classification
Fallback Responses: Graceful handling when the model can't classify inputs
Web Interface: Clean, user-friendly Gradio interface
Customizable Intents: Easy to modify responses and add new conversation patterns

🚀 Live Demo
Try the live chatbot: Hugging Face Spaces Link
📋 Requirements
gradio==4.44.0
torch>=1.9.0,<2.3.0
nltk>=3.7
numpy>=1.21.0
🛠️ Installation & Setup
1. Clone the Repository
bashgit clone https://github.com/YOUR_USERNAME/mental-health-chatbot.git
cd mental-health-chatbot
2. Install Dependencies
bashpip install -r requirements.txt
3. Train the Model (Optional)
If you want to retrain the model with your own data:
bashpython train_chatbot.py
4. Run the Application
bashpython app.py
The chatbot will be available at http://localhost:7860
📁 Project Structure
mental-health-chatbot/
├── app.py                 # Main Gradio application
├── train_chatbot.py       # Training script
├── intents.json          # Training data (patterns & responses)
├── chatbot_model.pth     # Trained PyTorch model
├── metadata.json         # Model metadata (vocabulary, intents)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore file
🧠 How It Works

Intent Classification: The model classifies user input into predefined intents (greeting, stress, anxiety, etc.)
Response Selection: Based on the classified intent, a relevant response is randomly selected
Fallback System: If confidence is low, the system uses pattern matching for basic responses
Continuous Learning: Easy to add new intents and retrain the model

🎯 Supported Intents

Greeting: Hello, hi, good morning, etc.
Goodbye: Bye, see you later, good night, etc.
Stress: Feeling overwhelmed, stressed, pressure, etc.
Anxiety: Panic attacks, worried, anxious feelings, etc.
Motivation: Need encouragement, feeling hopeless, etc.
Self-Care: Relaxation tips, wellness advice, etc.
Resources: Crisis support, helpline information, etc.
Loneliness: Feeling isolated, need connection, etc.
Sleep: Insomnia, sleep troubles, nightmares, etc.
Gratitude: Positive reinforcement, thankfulness, etc.

🔧 Customization
Adding New Intents

Edit intents.json to add new patterns and responses
Retrain the model: python train_chatbot.py
Restart the application

Modifying Responses
Simply edit the responses array in intents.json for any intent.
📊 Training Details

Model: 3-layer neural network (128, 64, output neurons)
Features: Bag of words representation
Preprocessing: NLTK tokenization and lemmatization
Training: 1000 epochs with Adam optimizer
Activation: ReLU with dropout for regularization

⚠️ Important Disclaimer
This chatbot is designed for supportive conversation and general wellness. It is not a replacement for professional mental health care. If you're experiencing a mental health crisis, please contact:

Emergency Services: Your local emergency number
Crisis Hotlines:

Vandrevala Foundation (India): 1860 266 2345
iCall (India): +91 9152987821
National Suicide Prevention Lifeline (US): 988



🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/new-intent)
Commit your changes (git commit -am 'Add new intent for motivation')
Push to the branch (git push origin feature/new-intent)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Built with Gradio for the web interface
NLTK for natural language processing
PyTorch for the neural network
Mental health resources from various wellness organizations

📞 Support
If you have questions or need help:

Open an issue on GitHub
Check the documentation
Contact: your-email@example.com


Remember: Taking care of your mental health is important. This chatbot is here to provide support, but please seek professional help when needed. 💙