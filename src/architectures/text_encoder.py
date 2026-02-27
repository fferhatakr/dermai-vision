from transformers import DistilBertForSequenceClassification

def build_nlp_model():
    #We download the model and set the number of output classes to Normal/Risk.
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 2,
        
    )
    
    return model