import torch
import torch.nn.functional as F
from PIL import Image
from src.architectures.vision_model import DermaScanModelV2
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torchvision.transforms as transforms



class DermatologistAI:

    #This class is used to load the trained models and make predictions.
    def __init__(self, cv_model_path, nlp_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cv_model = DermaScanModelV2()
        checkpoint = torch.load(cv_model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            
            name = k.replace("model.", "").replace("backbone.", "mobilenet_model.features.")
            name = name.replace("classifier.", "mobilenet_model.classifier.")
            new_state_dict[name] = v
        
        self.cv_model.to(self.device) 
        self.cv_model.eval()
        #We define the transformations to be applied to the images.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #We load the tokenizer and model for the NLP part.
        self.nlp_tokenizer = DistilBertTokenizer.from_pretrained(nlp_model_path)
        self.nlp_model = DistilBertForSequenceClassification.from_pretrained(nlp_model_path)
        self.nlp_model.to(self.device)
        self.nlp_model.eval()

        
    def analyze_image(self, image_path):
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad(): #We turn off the gradient calculation to speed up the process.
            outputs = self.cv_model(img_tensor)
            probs = F.softmax(outputs,dim=1)

        cv_risk_prob =(probs[0][0] + probs[0][1] + probs[0][4]).item() #We extract the probability of the first class (benign).
        normal_prob = (probs[0][2] + probs[0][3] + probs[0][5]).item()
        cv_risk_prob = 1.0 - normal_prob
        return cv_risk_prob

    def analyze_symptom(self, text):

        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad(): #We turn off the gradient calculation to speed up the process.
            outputs = self.nlp_model(**inputs)
            logits = outputs.logits
            probs=F.softmax(logits,dim=1)
            nlp_risk_prob = probs[0][1].item()
        return nlp_risk_prob

    #This function is used to combine the results of the two models.
    
    def hybrid_diagnosis(self, image_path, text, cv_weight=0.7, nlp_weight=0.3):
        cv_score = self.analyze_image(image_path)
        nlp_score = self.analyze_symptom(text)
        final_risk_score = cv_score*cv_weight +nlp_score*nlp_weight

        if final_risk_score >= 0.50:
            diagnosis = " RISKY (Consult a Specialist)"
        else:
            diagnosis = " NORMAL"
        
        return {
            "Image_Risk": cv_score,
            "Complaint_Risk": nlp_score,
            "Hybrid_Score": final_risk_score,
            "Diagnosis": diagnosis
        }

#This block is used to test the model.
if __name__ == "__main__":
    CV_PATH = "models/best_lightning_model-v4.ckpt"
    NLP_PATH = "models/nlp_v1"
    
    print(" DermaScan AI  Loading... Please wait.")
    ai_asistan = DermatologistAI(cv_model_path=CV_PATH, nlp_model_path=NLP_PATH)
    

    test_image = "Data/images/all_data/akiec/ISIC_0026149.jpg"  
    test_text = "The mark on my body spread very quickly and darkened in colour.."
    

    print("\n Analysis in progress...")
    result = ai_asistan.hybrid_diagnosis(image_path=test_image, text=test_text, cv_weight=0.7, nlp_weight=0.3)
    

    print("\n" + "="*40)
    print(f" Image Risk : %{result['Image_Risk']*100:.2f}")
    print(f" Complaint Risk : %{result['Complaint_Risk']*100:.2f}")
    print(f" HYBRID SCORE : %{result['Hybrid_Score']*100:.2f}")
    print(f" DIAGNOSIS : {result['Diagnosis']}")
    print("="*40)