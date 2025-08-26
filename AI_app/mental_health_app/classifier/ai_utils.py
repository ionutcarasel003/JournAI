import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#blala

model_path = r"D:\JournAI\AI_trainer\model"

# Use AutoTokenizer and AutoModelForSequenceClassification to automatically detect the correct model type
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

id2label = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
}

def predict_emotions(text):
    if model is None or tokenizer is None:
        print("Model not loaded, returning neutral")
        return "neutral", 0.0
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
        
        emotion_scores = [(id2label[i], probs[i] * 100) for i in range(len(probs))]
        emotion_scores.sort(key = lambda x: x[1], reverse = True)
        print(f"Predicted emotions: {emotion_scores}")
        return emotion_scores
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return "neutral", 0.0