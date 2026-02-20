from transformers import DistilBertForSequenceClassification

def build_nlp_model():
    # 1. Modeli indir ve çıkış sayısını Normal/Riskli
    model = DistilBertForSequenceClassification.from_pretrained(
        "dbmdz/distilbert-base-turkish-cased",
        num_labels = 2,
        
    )
    
    return model