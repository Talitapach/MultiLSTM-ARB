from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from keras_tuner import HyperModel

class MultiOutputLSTMHyperModel(HyperModel):
    def __init__(self, window_size, num_features, pred_steps):
        self.window_size = window_size
        self.num_features = num_features
        self.pred_steps = pred_steps

    def build(self, hp):
        K.clear_session()
        lstm_units = hp.Int('lstm_units', min_value=128, max_value=512, step=32)
        dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=32) 
        dropout_rate = hp.Float('dropout_rate', min_value=0.05, max_value=0.5, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        inputs = layers.Input(shape=(self.window_size, self.num_features))
        x = layers.LSTM(units=lstm_units)(inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        classification_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
        regression_output = layers.Dense(self.pred_steps, activation='linear', name='regression')(x)

        model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={'classification': 'binary_crossentropy', 'regression': 'mean_squared_error'},
            metrics={'classification': 'accuracy', 'regression': 'mae'}
        )

        return model



class MultiOutputLSTMBidirecionalHyperModel(HyperModel):
    def __init__(self, window_size, num_features, pred_steps):
        self.window_size = window_size
        self.num_features = num_features
        self.pred_steps = pred_steps

    def build(self, hp):
        lstm_units = hp.Int('lstm_units', min_value=128, max_value=512, step=32)
        dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.05, max_value=0.5, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        inputs = layers.Input(shape=(self.window_size, self.num_features))
        x = layers.Bidirectional(layers.LSTM(units=lstm_units))(inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        classification_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
        regression_output = layers.Dense(self.pred_steps, activation='linear', name='regression')(x)

        model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                      loss={'classification': 'binary_crossentropy', 'regression': 'mean_squared_error'},
                      metrics={'classification': 'accuracy', 'regression': 'mae'})

        return model