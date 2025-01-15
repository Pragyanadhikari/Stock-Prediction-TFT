# import os
# import joblib
# from keras.models import load_model
# import pandas as pd
# from test import code  # Ensure 'first_python_file' is the actual name of your first file

# def train_and_save_models(data_directory):
#     for file_name in os.listdir(data_directory):
#         if file_name.endswith('.csv'):
#             stock_name = os.path.splitext(file_name)[0]
#             file_path = os.path.join(data_directory, file_name)
#             print(f"Processing {stock_name}...")

#             model = code(file_path)

#             model_save_path = os.path.join(data_directory, f"{stock_name}_tft_model.h5")
#             model.save(model_save_path)
#             print(f"Model for {stock_name} saved at {model_save_path}")

#             scaler_save_path = os.path.join(data_directory, f"{stock_name}_scaler.pkl")
#             scaler = joblib.load('scaler.pkl')
#             joblib.dump(scaler, scaler_save_path)
#             print(f"Scaler for {stock_name} saved at {scaler_save_path}")

# if __name__ == "__main__":

#     data_directory = './csvfiles/'  
#     train_and_save_models(data_directory)


import os
import joblib
from keras.models import load_model
import pandas as pd
from test import code  # Ensure 'test' is the actual name of your first file

def train_and_save_models(data_directory):
    for file_name in os.listdir(data_directory):
        if file_name.endswith('.csv'):
            stock_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(data_directory, file_name)
            print(f"Processing {stock_name}...")

            # Train the model
            model = code(file_path)

            # Save the trained model
            model_save_path = os.path.join(data_directory, f"{stock_name}_tft_model.keras")
            model.save(model_save_path)
            print(f"Model for {stock_name} saved at {model_save_path}")

            # Save the scaler
            scaler_save_path = os.path.join(data_directory, f"{stock_name}_scaler.pkl")
            scaler = joblib.load('scaler.pkl')
            joblib.dump(scaler, scaler_save_path)
            print(f"Scaler for {stock_name} saved at {scaler_save_path}")

if __name__ == "__main__":
    data_directory = './csvfiles/'  # Change this to your actual directory path
    train_and_save_models(data_directory)
