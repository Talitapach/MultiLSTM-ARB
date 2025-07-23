import keras
from keras_tuner import RandomSearch

class HyperparameterTuner:
    def __init__(self, model_builder, x_train, y_train_class, y_train_regress, seed, epochs_tuning=50, validation_size=0.15):
        self.model_builder = model_builder
        self.x_train = x_train
        self.y_train_class = y_train_class
        self.y_train_regress = y_train_regress
        self.epochs_tuning = epochs_tuning
        self.seed = seed
        self.validation_size = validation_size

    def custom_loss(self, y_true_class, y_pred_class, y_true_regress, y_pred_regress, alpha=1.0, beta=1.0):
        loss_classification = keras.losses.binary_crossentropy(y_true_class, y_pred_class)
        loss_regression = keras.losses.mean_squared_error(y_true_regress, y_pred_regress)
        return alpha * loss_regression + beta * loss_classification

    def tune_and_validate(self, file_name):
        split_idx = int(len(self.x_train) * 0.8)

        X_train = self.x_train[:split_idx]
        X_val = self.x_train[split_idx:]
        y_train_class = self.y_train_class[:split_idx]
        y_val_class = self.y_train_class[split_idx:]
        y_train_regress = self.y_train_regress[:split_idx]
        y_val_regress = self.y_train_regress[split_idx:]


        tuner = RandomSearch(
            self.model_builder,
            objective="val_loss", 
            max_trials=30,  
            seed=self.seed,
            executions_per_trial=1,
            directory="hyperparam_tuning",
            project_name="dengue_surtos_lstm",
            overwrite=True
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )

        tuner.search(
            X_train,
            {"classification": y_train_class, "regression": y_train_regress},
            validation_data=(X_val, {"classification": y_val_class, "regression": y_val_regress}),
            epochs=self.epochs_tuning,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )

        best_model = tuner.get_best_models(1)[0]
        best_model.save(file_name)
        return best_model