
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split, KFold
# from tensorflow import keras
# from tensorflow.keras import layers, models
# import keras_tuner as kt
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam, Adamax
# import joblib
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from keras.models import load_model
# from keras.utils import plot_model
# import time

# def code(outside_path):

#     df_stock = pd.read_csv(outside_path, parse_dates=['Date'], dayfirst=True)
#     df_stock['Open'] = df_stock['Open'].astype(str)
#     df_stock['Open'] = df_stock['Open'].str.replace(',', '').astype(float)
#     df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%d/%m/%Y')
#     df_stock['day_of_week'] = df_stock['Date'].dt.dayofweek
#     df_stock['month'] = df_stock['Date'].dt.month

#     features = ['Date', 'Open', 'day_of_week', 'month']
#     df_stock = df_stock[features]

#     scaler = MinMaxScaler()
#     df_scaled = pd.DataFrame(scaler.fit_transform(df_stock[['Open', 'day_of_week', 'month']]), columns=['Open', 'day_of_week', 'month'])
#     joblib.dump(scaler, 'scaler.pkl')

#     def df_to_X_y(df, window_size=5):
#         df_as_np = df.to_numpy()
#         X = []
#         y = []
#         for i in range(len(df_as_np) - window_size):
#             X.append(df_as_np[i:i + window_size])
#             y.append(df_as_np[i + window_size][0])
#         return np.array(X), np.array(y)
    
#     WINDOW_SIZE = 6
#     X, y = df_to_X_y(df_scaled, WINDOW_SIZE)

#     def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
#         x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#         x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#         x = layers.Dropout(dropout)(x)
#         res = x + inputs

#         x = layers.LayerNormalization(epsilon=1e-6)(res)
#         x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#         x = layers.Dropout(dropout)(x)
#         x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#         return x + res

#     def gated_residual_network(inputs, units): 
#         x = layers.Dense(units, activation='relu')(inputs) 
#         x = layers.Dense(inputs.shape[-1])(x) 
#         gate = layers.Dense(inputs.shape[-1], activation='sigmoid')(inputs) 
#         return x * gate + inputs

#     def build_tft_model(hp):
#         input_shape = (WINDOW_SIZE, 3)
#         inputs = layers.Input(shape=input_shape)
#         x = inputs
#         x = gated_residual_network(x, units=hp.Int('grn_units', min_value=32, max_value=128, step=32))
#         x = layers.LSTM(units=50, return_sequences=True)(x)
#         x = layers.LSTM(units=50, return_sequences=True)(x)
#         for i in range(hp.Int('num_transformer_blocks', 2, 8, 2)):
#             x = transformer_encoder(
#                 x,
#                 head_size=hp.Int('head_size', 8, 256, 32),
#                 num_heads=hp.Int('num_heads', 2, 16),
#                 ff_dim=hp.Int('ff_dim', 4, 64),
#                 dropout=hp.Float(f'dropout_{i}', 0.1, 0.6)
#             )
#         x = layers.GlobalAveragePooling1D()(x)
#         for i in range(hp.Int('num_mlp_layers', 1, 3)):
#             x = layers.Dense(hp.Int(f'mlp_units_{i}', 32, 256, 32))(x)
#             x = layers.Activation('relu')(x)
#             x = layers.Dropout(hp.Float(f'mlp_dropout_{i}', 0.1, 0.6))(x)
#         outputs = layers.Dense(1)(x)
#         model = models.Model(inputs, outputs)
#         optimizer_name = hp.Choice('optimizer', ['adam', 'adamax'])
#         learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
#         if optimizer_name == 'adam':
#             optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#         else:
#             optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
#         model.compile(
#             optimizer=optimizer, 
#             loss='mean_absolute_error', 
#             metrics=['mean_absolute_error', 'mean_squared_error']
#         )
#         return model

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
#     tuner = kt.RandomSearch(
#         build_tft_model,
#         objective='val_loss',
#         max_trials=30,
#         directory='./tft_tuning',
#         project_name='tft_project'
#     )
#     tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#     model = build_tft_model(best_hps)

#     def plot_loss(history):
#         plt.figure(figsize=(10, 6))
#         plt.plot(history.history['loss'], label='Training Loss')
#         plt.plot(history.history['val_loss'], label='Validation Loss')
#         plt.title('Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.show()

#     def plot_mae(history):
#         plt.figure(figsize=(10, 6))
#         if 'mean_absolute_error' in history.history and 'val_mean_absolute_error' in history.history:
#             plt.plot(history.history['mean_absolute_error'], label='Training MAE')
#             plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
#             plt.title('Training and Validation MAE')
#             plt.xlabel('Epochs')
#             plt.ylabel('MAE')
#             plt.legend()
#             plt.show()
#         else:
#             print("MAE metric not found in history.")

#     def plot_mse(history):
#         plt.figure(figsize=(10, 6))
#         if 'mean_squared_error' in history.history and 'val_mean_squared_error' in history.history:
#             plt.plot(history.history['mean_squared_error'], label='Training MSE')
#             plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
#             plt.title('Training and Validation MSE')
#             plt.xlabel('Epochs')
#             plt.ylabel('MSE')
#             plt.legend()
#             plt.show()
#         else:
#             print("MSE metric not found in history.")
    
#     best_model_hyperparameters = None
#     best_model_mae = float('inf')
#     worst_model_hyperparameters = None
#     worst_model_mae = float('-inf')

#     nested_scores = []
#     outer_cv = KFold(n_splits=5, shuffle=True, random_state=100)

#     for fold_outer_idx, (train_ix, test_ix) in enumerate(outer_cv.split(X)):
#         X_train_outer, X_test_outer = X[train_ix], X[test_ix]
#         y_train_outer, y_test_outer = y[train_ix], y[test_ix]
#         inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
#         for fold_idx, (train_ix_inner, val_ix) in enumerate(inner_cv.split(X_train_outer)):
#             X_train_inner, X_val = X_train_outer[train_ix_inner], X_train_outer[val_ix]
#             y_train_inner, y_val = y_train_outer[train_ix_inner], y_train_outer[val_ix]
#             tuner = kt.RandomSearch(
#                 build_tft_model,
#                 objective='val_loss',
#                 max_trials=5,
#                 directory=f'./keras_tuner_random_dir_fold_tf_{fold_outer_idx}_{fold_idx}',
#                 project_name=f'hyperparameter_random_tuning_fold_tf_{fold_outer_idx}_{fold_idx}'
#             )
#             tuner.search(X_train_inner, y_train_inner, validation_data=(X_val, y_val), epochs=5)
#             best_hps = tuner.oracle.get_best_trials(1)[0].hyperparameters
#             print(f"Best hyperparameters for fold {fold_idx}: {best_hps}")
#             model = build_tft_model(best_hps)
#             es = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
#             history = model.fit(X_train_inner, y_train_inner, validation_data=(X_val, y_val), epochs=5, batch_size=32, callbacks=es)
#             plot_loss(history)
#             plot_mae(history)
#             plot_mse(history)
#             y_pred = model.predict(X_val)
#             mae = mean_absolute_error(y_val, y_pred)
#             if mae < best_model_mae:
#                 best_model_mae = mae
#                 best_model_hyperparameters = best_hps
#                 best_model = model
#             if mae > worst_model_mae:
#                 worst_model_mae = mae
#                 worst_model_hyperparameters = best_hps
#                 worst_model = model
#         start_time = time.time()
#         y_pred_best = best_model.predict(X_test_outer)
#         y_pred_worst = worst_model.predict(X_test_outer)
#         end_time = time.time()
#         mse_best = mean_squared_error(y_test_outer, y_pred_best)
#         mae_best = mean_absolute_error(y_test_outer, y_pred_best)
#         rmse_best = np.sqrt(mse_best)
#         r2_best = r2_score(y_test_outer, y_pred_best)
#         time_duration = end_time - start_time
#         mse_worst = mean_squared_error(y_test_outer, y_pred_worst)
#         mae_worst = mean_absolute_error(y_test_outer, y_pred_worst)
#         rmse_worst = np.sqrt(mse_worst)
#         r2_worst = r2_score(y_test_outer, y_pred_worst)
#         nested_scores.append({
#             "Best Model": {
#                 "MSE": mse_best,
#                 "R^2": r2_best,
#                 "RMSE": rmse_best,
#                 "MAE": mae_best,
#                 "testing time": time_duration
#             },
#             "Worst Model": {
#                 "MSE": mse_worst,
#                 "R^2": r2_worst,
#                 "RMSE": rmse_worst,
#                 "MAE": mae_worst
#             }
#         })

#     print("Nested Cross-Validation Scores:")
#     for idx, score in enumerate(nested_scores):
#         print(f"Fold {idx+1}:")
#         print(f"  Best Model - MSE: {score['Best Model']['MSE']}, R^2: {score['Best Model']['R^2']}, RMSE: {score['Best Model']['RMSE']}, MAE: {score['Best Model']['MAE']}, Testing Time: {score['Best Model']['testing time']}")
#         print(f"  Worst Model - MSE: {score['Worst Model']['MSE']}, R^2: {score['Worst Model']['R^2']}, RMSE: {score['Worst Model']['RMSE']}, MAE: {score['Worst Model']['MAE']}")

#     avg_best_mse = np.mean([score['Best Model']['MSE'] for score in nested_scores])
#     avg_best_mae = np.mean([score['Best Model']['MAE'] for score in nested_scores])
#     avg_best_r2 = np.mean([score['Best Model']['R^2'] for score in nested_scores])
#     avg_best_rmse = np.mean([score['Best Model']['RMSE'] for score in nested_scores])

#     print("\nAverage Best Model Metrics across all folds:")
#     print(f"Average MSE: {avg_best_mse}")
#     print(f"Average MAE: {avg_best_mae}")
#     print(f"Average R^2: {avg_best_r2}")
#     print(f"Average RMSE: {avg_best_rmse}")

#     y_pred = best_model.predict(X).flatten()
#     y_observed = y.flatten()

#     plt.figure(figsize=(4, 4))
#     plt.scatter(y_observed, y_pred, color='blue', alpha=0.5)
#     plt.plot([min(y_observed), max(y_observed)], [min(y_observed), max(y_observed)], color='red', linestyle='--')
#     plt.title('Transformer - Observed vs Predicted')
#     plt.xlabel('Observed Ltp')
#     plt.ylabel('Predicted Ltp')
#     plt.grid(True)
#     plt.show()

#     optimizer = best_model.optimizer
#     learning_rate = float(optimizer.learning_rate.numpy())

#     print("Optimizer:", type(optimizer).__name__)
#     print("Learning Rate:", learning_rate)

#     def get_dropout_rate(layer):
#         if hasattr(layer, 'rate'):
#             return layer.rate
#         elif hasattr(layer, 'dropout'):
#             return layer.dropout
#         else:
#             return None

#     dropout_rates = []

#     for layer in best_model.layers:
#         rate = get_dropout_rate(layer)
#         if rate is not None:
#             dropout_rates.append((layer.name, rate))

#     print("Dropout Rates:")
#     for layer_name, rate in dropout_rates:
#         print(f"{layer_name}: {rate}")

#     best_hyperparameters_dict = best_model_hyperparameters.values
#     worst_hyperparameters_dict = worst_model_hyperparameters.values

#     print("Best Hyperparameters:", best_hyperparameters_dict)
#     print("Worst Hyperparameters:", worst_hyperparameters_dict)

#     best_model.save('./STOCKtft.h5')

#     best_model = load_model('./STOCKtft.h5')

#     print(best_model.summary())

#     plot_model(best_model, to_file='stocktft.png', show_shapes=True, show_layer_names=True, rankdir='TB')

#     return best_model


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from tensorflow.keras import layers, models
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Adamax
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def code(outside_path):

    df_stock = pd.read_csv(outside_path, parse_dates=['Date'], dayfirst=True)
    df_stock['Open'] = df_stock['Open'].astype(str)
    df_stock['Open'] = df_stock['Open'].str.replace(',', '').astype(float)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%d/%m/%Y')
    df_stock['day_of_week'] = df_stock['Date'].dt.dayofweek
    df_stock['month'] = df_stock['Date'].dt.month

    features = ['Date', 'Open', 'day_of_week', 'month']
    df_stock = df_stock[features]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_stock[['Open', 'day_of_week', 'month']]), columns=['Open', 'day_of_week', 'month'])
    joblib.dump(scaler, 'scaler.pkl')

    def df_to_X_y(df, window_size=5):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np) - window_size):
            X.append(df_as_np[i:i + window_size])
            y.append(df_as_np[i + window_size][0])
        return np.array(X), np.array(y)
    
    WINDOW_SIZE = 6
    X, y = df_to_X_y(df_scaled, WINDOW_SIZE)

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def gated_residual_network(inputs, units): 
        x = layers.Dense(units, activation='relu')(inputs) 
        x = layers.Dense(inputs.shape[-1])(x) 
        gate = layers.Dense(inputs.shape[-1], activation='sigmoid')(inputs) 
        return x * gate + inputs

    def build_tft_model(hp):
        input_shape = (WINDOW_SIZE, 3)
        inputs = layers.Input(shape=input_shape)
        x = inputs
        x = gated_residual_network(x, units=hp.Int('grn_units', min_value=32, max_value=128, step=32))
        x = layers.LSTM(units=50, return_sequences=True)(x)
        x = layers.LSTM(units=50, return_sequences=True)(x)
        for i in range(hp.Int('num_transformer_blocks', 2, 8, 2)):
            x = transformer_encoder(
                x,
                head_size=hp.Int('head_size', 8, 256, 32),
                num_heads=hp.Int('num_heads', 2, 16),
                ff_dim=hp.Int('ff_dim', 4, 64),
                dropout=hp.Float(f'dropout_{i}', 0.1, 0.6)
            )
        x = layers.GlobalAveragePooling1D()(x)
        for i in range(hp.Int('num_mlp_layers', 1, 3)):
            x = layers.Dense(hp.Int(f'mlp_units_{i}', 32, 256, 32))(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(hp.Float(f'mlp_dropout_{i}', 0.1, 0.6))(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs, outputs)
        optimizer_name = hp.Choice('optimizer', ['adam', 'adamax'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='mean_absolute_error', 
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        return model

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
    tuner = kt.RandomSearch(
        build_tft_model,
        objective='val_loss',
        max_trials=30,
        directory='./tft_tuning',
        project_name='tft_project'
    )
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_tft_model(best_hps)

    def plot_loss(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_mae(history):
        plt.figure(figsize=(10, 6))
        if 'mean_absolute_error' in history.history and 'val_mean_absolute_error' in history.history:
            plt.plot(history.history['mean_absolute_error'], label='Training MAE')
            plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
            plt.title('Training and Validation MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.show()
        else:
            print("MAE metric not found in history.")

    def plot_mse(history):
        plt.figure(figsize=(10, 6))
        if 'mean_squared_error' in history.history and 'val_mean_squared_error' in history.history:
            plt.plot(history.history['mean_squared_error'], label='Training MSE')
            plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
            plt.title('Training and Validation MSE')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
        else:
            print("MSE metric not found in history.")
    
    best_model_hyperparameters = None
    best_model_mae = float('inf')
    worst_model_hyperparameters = None
    worst_model_mae = float('-inf')

    nested_scores = []
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=100)

    for fold_outer_idx, (train_ix, test_ix) in enumerate(outer_cv.split(X)):
        X_train_outer, X_test_outer = X[train_ix], X[test_ix]
        y_train_outer, y_test_outer = y[train_ix], y[test_ix]
        inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
        for fold_idx, (train_ix_inner, val_ix) in enumerate(inner_cv.split(X_train_outer)):
            X_train_inner, X_val = X_train_outer[train_ix_inner], X_train_outer[val_ix]
            y_train_inner, y_val = y_train_outer[train_ix_inner], y_train_outer[val_ix]
            tuner = kt.RandomSearch(
                build_tft_model,
                objective='val_loss',
                max_trials=5,
                directory=f'./keras_tuner_random_dir_fold_tf_{fold_outer_idx}_{fold_idx}',
                project_name=f'hyperparameter_random_tuning_fold_tf_{fold_outer_idx}_{fold_idx}'
            )
            tuner.search(X_train_inner, y_train_inner, validation_data=(X_val, y_val), epochs=5)
            best_hps = tuner.oracle.get_best_trials(1)[0].hyperparameters
            print(f"Best hyperparameters for fold {fold_idx}: {best_hps}")
            model = build_tft_model(best_hps)
            es = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            history = model.fit(X_train_inner, y_train_inner, validation_data=(X_val, y_val), epochs=5, batch_size=32, callbacks=es)
            plot_loss(history)
            plot_mae(history)
            plot_mse(history)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            if mae < best_model_mae:
                best_model_mae = mae
                best_model_hyperparameters = best_hps
                best_model = model
            if mae > worst_model_mae:
                worst_model_mae = mae
                worst_model_hyperparameters = best_hps
                worst_model = model
                start_time = time.time()
        y_pred_best = best_model.predict(X_test_outer)
        y_pred_worst = worst_model.predict(X_test_outer)
        end_time = time.time()
        mse_best = mean_squared_error(y_test_outer, y_pred_best)
        mae_best = mean_absolute_error(y_test_outer, y_pred_best)
        rmse_best = np.sqrt(mse_best)
        r2_best = r2_score(y_test_outer, y_pred_best)
        time_duration = end_time - start_time
        mse_worst = mean_squared_error(y_test_outer, y_pred_worst)
        mae_worst = mean_absolute_error(y_test_outer, y_pred_worst)
        rmse_worst = np.sqrt(mse_worst)
        r2_worst = r2_score(y_test_outer, y_pred_worst)
        nested_scores.append({
            "Best Model": {
                "MSE": mse_best,
                "R^2": r2_best,
                "RMSE": rmse_best,
                "MAE": mae_best,
                "testing time": time_duration
            },
            "Worst Model": {
                "MSE": mse_worst,
                "R^2": r2_worst,
                "RMSE": rmse_worst,
                "MAE": mae_worst
            }
        })

    print("Nested Cross-Validation Scores:")
    for idx, score in enumerate(nested_scores):
        print(f"Fold {idx+1}:")
        print(f"  Best Model - MSE: {score['Best Model']['MSE']}, R^2: {score['Best Model']['R^2']}, RMSE: {score['Best Model']['RMSE']}, MAE: {score['Best Model']['MAE']}, Testing Time: {score['Best Model']['testing time']}")
        print(f"  Worst Model - MSE: {score['Worst Model']['MSE']}, R^2: {score['Worst Model']['R^2']}, RMSE: {score['Worst Model']['RMSE']}, MAE: {score['Worst Model']['MAE']}")

    avg_best_mse = np.mean([score['Best Model']['MSE'] for score in nested_scores])
    avg_best_mae = np.mean([score['Best Model']['MAE'] for score in nested_scores])
    avg_best_r2 = np.mean([score['Best Model']['R^2'] for score in nested_scores])
    avg_best_rmse = np.mean([score['Best Model']['RMSE'] for score in nested_scores])

    print("\nAverage Best Model Metrics across all folds:")
    print(f"Average MSE: {avg_best_mse}")
    print(f"Average MAE: {avg_best_mae}")
    print(f"Average R^2: {avg_best_r2}")
    print(f"Average RMSE: {avg_best_rmse}")

    y_pred = best_model.predict(X).flatten()
    y_observed = y.flatten()

    plt.figure(figsize=(4, 4))
    plt.scatter(y_observed, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_observed), max(y_observed)], [min(y_observed), max(y_observed)], color='red', linestyle='--')
    plt.title('Transformer - Observed vs Predicted')
    plt.xlabel('Observed Ltp')
    plt.ylabel('Predicted Ltp')
    plt.grid(True)
    plt.show()

    optimizer = best_model.optimizer
    learning_rate = float(optimizer.learning_rate.numpy())

    print("Optimizer:", type(optimizer).__name__)
    print("Learning Rate:", learning_rate)

    def get_dropout_rate(layer):
        if hasattr(layer, 'rate'):
            return layer.rate
        elif hasattr(layer, 'dropout'):
            return layer.dropout
        else:
            return None

    dropout_rates = []

    for layer in best_model.layers:
        rate = get_dropout_rate(layer)
        if rate is not None:
            dropout_rates.append((layer.name, rate))

    print("Dropout Rates:")
    for layer_name, rate in dropout_rates:
        print(f"{layer_name}: {rate}")

    best_hyperparameters_dict = best_model_hyperparameters.values
    worst_hyperparameters_dict = worst_model_hyperparameters.values

    print("Best Hyperparameters:", best_hyperparameters_dict)
    print("Worst Hyperparameters:", worst_hyperparameters_dict)

    return best_model
