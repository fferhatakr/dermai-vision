import torch 
import torch.nn.functional as F 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification 

path = "models/nlp_v1"
tokenizer = DistilBertTokenizer.from_pretrained(path)
model = DistilBertForSequenceClassification.from_pretrained(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs= F.softmax(logits,dim=1)
        prediction = torch.argmax(probs,dim = 1).item()

    return prediction,probs[0]

line = "Vücudumdaki leke çok kısa sürede yayıldı ve rengi koyulaştı."
result, probability = predict(line)

label = "⚠️ RISKY" if result == 1 else "✅ NORMAL"
print(f"Result: {label}")
print(f"Normal Probability: %{probability[0]*100:.2f} | Probability of Risk: %{probability[1]*100:.2f}")
