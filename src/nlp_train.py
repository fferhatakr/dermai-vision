import pandas as pd
from torch.utils.data import DataLoader
from nlp_dataset import SymptomDataset
from transformers import DistilBertTokenizer
from transformers import DataCollatorWithPadding
from nlp_model import build_nlp_model
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split






EPOCH = 5

df = pd.read_csv('Data/symptoms.csv')
tokenizer = DistilBertTokenizer.from_pretrained('dbmdz/distilbert-base-turkish-cased')

text = df['texts'].tolist()
label = df['labels'].tolist()

total_dataset = len(df)
random_state=42

collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    text, 
    label, 
    test_size=0.20, 
    random_state=42
)

train_dataset = SymptomDataset(train_texts,train_labels,tokenizer)
val_dataset = SymptomDataset(val_texts,val_labels,tokenizer)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    collate_fn=collator,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    collate_fn=collator,
    shuffle=False
)


model = build_nlp_model()
optimizer = AdamW(model.parameters(),lr=2e-5)


for i in range(EPOCH):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    model.eval()
    total_val_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()     
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == batch['labels'])


    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct_predictions.double() / len(val_dataset)

    print(f"Epoch {i+1}/{EPOCH}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: %{accuracy*100:.2f}")

# En son modeli kaydet
model.save_pretrained("models/nlp_v1")
tokenizer.save_pretrained("models/nlp_v1")

