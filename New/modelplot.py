from keras.models import load_model
from keras.utils import plot_model

best_model = load_model('./csvfiles/NULBdata_tft_model.keras')
print(best_model.summary())

# Visualize the model architecture and save the diagram to a file
plot_model(best_model, to_file='NULBstocktft.png', show_shapes=True, show_layer_names=True, rankdir='TB')