from transformers import DistilBertForSequenceClassification


def build_nlp_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 2,
        
    )
    
    return model