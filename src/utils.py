import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def matris_cizdir(gercek_etiketler, modelin_tahminleri, baslik="Karmaşıklık Matrisi"):
  
    cm = confusion_matrix(gercek_etiketler, modelin_tahminleri)

  
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
    plt.ylabel('Gerçek Değer (Doktor Ne Dedi?)')
    plt.title('Cepteki Dermatolog V3 (Adaletli) - Karmaşıklık Matrisi')
    plt.show()