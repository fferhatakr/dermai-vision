#Import the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


#When we want to draw a matrix, it will suffice to call the matrix_draw function.
def matrix_draw(real_labels, model_predictions, title="Complexity Matrix"):
  
    cm = confusion_matrix(real_labels, model_predictions)

  
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
    plt.ylabel('True Value (What Did the Doctor Say?)')
    plt.title(title)
    plt.show()