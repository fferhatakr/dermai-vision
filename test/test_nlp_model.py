import torch
from src.models.nlp_model import build_nlp_model
from transformers import DistilBertTokenizer

def test_nlp_model_output():
    """
    This test checks whether our NLP model can successfully read the patient's complaint text
    and convert it into a mathematical Tensor.
    """


    model = build_nlp_model()
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    fake_text = "The wound area has started to burn a lot for about two days now."
    inputs = tokenizer(fake_text, return_tensors="pt")
    with torch.no_grad():
        result = model(**inputs)
    
    real_result = result.logits
    assert isinstance(real_result,torch.Tensor), f"The model's output is not a Tensor!"
    excapted_shape = torch.Size([1,2])
    assert real_result.shape == excapted_shape ,  f"NLP model size is incorrect! Output: {real_result.shape}"




