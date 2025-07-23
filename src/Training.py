import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, mean_absolute_error, 
    mean_squared_error, roc_curve, auc, accuracy_score, roc_auc_score, classification_report, confusion_matrix
)


class MultiTaskModelTrainer:
    def __init__(self, model, X_train, y_train_class, y_train_regress, 
                 X_test, y_test_class, y_test_regress, timestamp):
        self.model = model
        self.X_train = X_train
        self.y_train_class = y_train_class
        self.y_train_regress = y_train_regress
        self.X_test = X_test
        self.y_test_class = y_test_class
        self.y_test_regress = y_test_regress
        self.timestamp = timestamp 

        self.history = None
        self.y_pred_class = None
        self.y_pred_reg = None
        self.test_results = None


    def compile_model(self, classification_loss_weight=1.0, regression_loss_weight=1.0):
        self.model.compile(
            optimizer='adam',
            loss={
                'classification': 'binary_crossentropy',  
                'regression': 'mean_squared_error'                      
            },
            metrics={
                'classification': ['accuracy'],
                'regression': ['mae']                    
            }
        )

    def train(self, epochs=100, batch_size=32, n_splits=5, bootstrap_samples=1000, confidence_levels=[95]):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        f1_scores = [] 
        auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_train, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_train_class, y_val_class = self.y_train_class[train_idx], self.y_train_class[val_idx]
            y_train_regress, y_val_regress = self.y_train_regress[train_idx], self.y_train_regress[val_idx]
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

            history = self.model.fit(
                X_train,
                {'classification': y_train_class, 'regression': y_train_regress},
                validation_data=(X_val, {'classification': y_val_class, 'regression': y_val_regress}),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stopping, reduce_lr]  
            )

            y_val_pred = self.model.predict(X_val)

            y_val_pred_class = y_val_pred[0] 

            y_pred_class_binary = (y_val_pred_class > 0.5).astype(int)

            f1 = f1_score(y_val_class, y_pred_class_binary)
            auc = roc_auc_score(y_val_class, y_val_pred_class)

            f1_scores.append(f1)
            auc_scores.append(auc)



    def evaluate(self):
        self.test_results = self.model.evaluate(self.X_test, {'classification': self.y_test_class, 'regression': self.y_test_regress})
        print(f"Resultados no teste: {self.test_results}")

        self.y_pred = self.model.predict(self.X_test)

        y_pred_class = self.y_pred[0]  
        y_pred_reg = self.y_pred[1]  

        self.y_pred_class = (y_pred_class > 0.5).astype(int)

        print("Relatório de Classificação:")
        print(classification_report(self.y_test_class, self.y_pred_class))

        print("Matriz de Confusão:")
        print(confusion_matrix(self.y_test_class, self.y_pred_class))

        return self.y_test_class, self.y_test_regress, y_pred_class, y_pred_reg


    def save_predictions(self):
        print(f"Tamanho de timestamp: {len(self.timestamp)}")
        print(f"Tamanho de y_test_class: {len(self.y_test_class)}")
        print(f"Tamanho de y_pred_class: {len(self.y_pred_class)}")
        print(f"Tamanho de y_test_regress: {len(self.y_test_regress)}")
        print(f"Tamanho de y_pred_reg: {len(self.y_pred_reg)}")

        if len(self.timestamp) == len(self.y_test_class) == len(self.y_pred_class) == len(self.y_test_regress) == len(self.y_pred_reg):
            predictions_df = pd.DataFrame({
                'timestamp': self.timestamp,
                'real_classification': self.y_test_class,
                'predicted_classification': self.y_pred_class,
                'real_regression': self.y_test_regress,
                'predicted_regression': self.y_pred_reg
            })

            predictions_df.to_csv(os.path.join(self.predictions_dir, 'predictions.csv'), index=False)

            summary_df = pd.DataFrame({
                'test_results': self.test_results,
                'accuracy': self.test_results[1],
                'mae': self.test_results[2],
                'mse': self.test_results[0]
            }, index=[0])
            summary_df.to_csv(os.path.join(self.summary_dir, 'summary.csv'), index=False)
        else:
            raise ValueError(" ")



    def plot_predictions(self, scaler):
        y_test_class = self.y_test_class
        y_test_reg = self.y_test_regress

        y_pred_reg = self.y_pred[1]
        y_pred_class = self.y_pred[0]

        timestamp_test = self.timestamp[-len(y_test_reg):]

        timestamp_test = np.array(timestamp_test)
        y_test_reg = np.array(y_test_reg)
        y_pred_reg = np.array(y_pred_reg)
        y_pred_class = np.array(y_pred_class).flatten() 

        y_pred_reg = y_pred_reg[-len(y_test_reg):]

        if len(y_pred_reg.shape) > 1:
            y_pred_reg = y_pred_reg[:, 0]

        y_test_reg_desescalonado = scaler.inverse_transform(y_test_reg.reshape(-1, 1)).flatten()
        y_pred_reg_desescalonado = scaler.inverse_transform(y_pred_reg.reshape(-1, 1)).flatten()

        y_pred_reg_desescalonado = np.maximum(y_pred_reg_desescalonado, 0)


        metrics = self.calculate_metrics(y_test_class, y_pred_class, y_test_reg_desescalonado, y_pred_reg_desescalonado)

        self.save_metrics_to_csv(metrics)

        fpr, tpr, roc_auc = metrics[7], metrics[8], metrics[9]
        self.plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png')

        plt.figure(figsize=(14, 8))
        plt.plot(timestamp_test, y_test_reg_desescalonado, label='Official (Number of Cases)', color='blue', linewidth=2)
        plt.plot(timestamp_test, y_pred_reg_desescalonado, label='Forecast (Number of Cases)', color='orange', linewidth=2)

        surtos_previstos = timestamp_test[y_pred_class >= 0.9]  
        surtos_valores = y_pred_reg_desescalonado[y_pred_class >= 0.9]  
        plt.scatter(surtos_previstos, surtos_valores, color='red', label='Outbreak Forecast', s=50)

        plt.ylabel('Number of Cases', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()



    def plot_predictions_with_historical(self, scaler, df_historical):
        y_test_class = self.y_test_class
        y_test_reg = self.y_test_regress

        y_pred_reg = self.y_pred[1] 
        y_pred_class = self.y_pred[0] 

        timestamp_test = self.timestamp[-len(y_test_reg):]

        timestamp_test = np.array(timestamp_test)
        y_test_reg = np.array(y_test_reg)
        y_pred_reg = np.array(y_pred_reg)
        y_pred_class = np.array(y_pred_class).flatten()  

        y_pred_reg = y_pred_reg[-len(y_test_reg):]

        if len(y_pred_reg.shape) > 1:
            y_pred_reg = y_pred_reg[:, 0]

        y_test_reg_desescalonado = scaler.inverse_transform(y_test_reg.reshape(-1, 1)).flatten()
        y_pred_reg_desescalonado = scaler.inverse_transform(y_pred_reg.reshape(-1, 1)).flatten()

        y_pred_reg_desescalonado = np.maximum(y_pred_reg_desescalonado, 0)

        plt.figure(figsize=(14, 8))

        plt.plot(df_historical['DT_NOTIFIC'], df_historical['NUM_CASOS_MES'], 
                color='black', label='Number of Cases', linewidth=1.5, linestyle='--')

        surtos_historicos = df_historical[df_historical['SURTO'] == 1]  
        plt.scatter(surtos_historicos['DT_NOTIFIC'], surtos_historicos['NUM_CASOS_MES'], 
                    color='purple', label='Outbreak Forecast', s=50)

        plt.plot(timestamp_test, y_test_reg_desescalonado, label='Official (Number of Cases)', color='blue', linewidth=2)

        plt.plot(timestamp_test, y_pred_reg_desescalonado, label='Forecast (Number of Cases)', color='orange', linewidth=2)

        surtos_previstos = timestamp_test[y_pred_class >= 0.9]  
        surtos_valores = y_pred_reg_desescalonado[y_pred_class >= 0.9]  
        plt.scatter(surtos_previstos, surtos_valores, color='red', label='Outbreak Forecast', s=50)

        plt.ylabel('Number of Cases', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.tight_layout()
        plt.xticks(rotation=45)

        plt.show()


    def save_predictions(self):
        if self.y_pred_class is not None and self.y_pred_reg is not None:
            y_pred_class = self.y_pred_class.flatten() if len(self.y_pred_class.shape) > 1 else self.y_pred_class
            y_pred_reg = self.y_pred_reg.flatten() if len(self.y_pred_reg.shape) > 1 else self.y_pred_reg
            
            predictions_df = pd.DataFrame({
                'y_test_class': self.y_test_class,
                'y_pred_class': y_pred_class,
                'y_test_regress': self.y_test_regress,
                'y_pred_reg': y_pred_reg
            })
            
            predictions_df.to_csv(os.path.join(self.predictions_dir, f'predictions.csv'), index=False)
            print("'predictions.csv'")
        else:
            print(" ")


    def calculate_metrics(self, y_true_class, y_pred_class, y_true_reg, y_pred_reg):
        if len(y_pred_reg.shape) > 1:
            y_pred_reg = y_pred_reg[:, 0] 

        y_pred_class_binary = (y_pred_class > 0.5).astype(int)
        precision = precision_score(y_true_class, y_pred_class_binary)
        recall = recall_score(y_true_class, y_pred_class_binary)
        f1 = f1_score(y_true_class, y_pred_class_binary)

        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
        roc_auc = auc(fpr, tpr)

        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        mse = mean_squared_error(y_true_reg, y_pred_reg)
        rmse = np.sqrt(mse)  
        medape = np.median(np.abs((y_true_reg - y_pred_reg) / y_true_reg)) * 100 

        mape = np.mean(np.abs((y_true_reg - y_pred_reg) / y_true_reg)) * 100  

        return precision, recall, f1, mae, mse, rmse, medape, fpr, tpr, roc_auc, mape



    def save_metrics_to_csv(self, metrics, file_path='metrics_results.csv'):
        metrics_df = pd.DataFrame([{
            'Precision': metrics[0],
            'Recall': metrics[1],
            'F1-Score': metrics[2],
            'MAE': metrics[3],
            'MSE': metrics[4],
            'RMSE': metrics[5],
            'MedAPE': metrics[6],
            'MAPE': metrics[10],
            'ROC AUC': metrics[9]  
        }])

        print("Metrics Results:")
        print(metrics_df)

 
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()

        plt.show()



    def bootstrap_evaluation(self, y_true_class, y_pred_class, y_true_reg, y_pred_reg, n_iterations=1000):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_scores = []

        mae_scores = []
        mse_scores = []
        rmse_scores = []
        medape_scores = []
        mape_scores = []

        for _ in range(n_iterations):
            indices = np.random.choice(len(y_true_class), size=len(y_true_class), replace=True)

            y_true_class_resampled = y_true_class[indices]
            y_pred_class_resampled = y_pred_class[indices]
            y_true_reg_resampled = y_true_reg[indices]
            y_pred_reg_resampled = y_pred_reg[indices]

            precision, recall, f1, mae, mse, rmse, medape, fpr, tpr, roc_auc, mape= self.calculate_metrics(
                y_true_class_resampled, y_pred_class_resampled, y_true_reg_resampled, y_pred_reg_resampled
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            roc_scores.append(roc_auc)

            mae_scores.append(mae)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            medape_scores.append(medape)
            mape_scores.append(mape)

        metrics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'roc_auc': roc_scores,
            'mae': mae_scores,
            'mse': mse_scores,
            'rmse': rmse_scores,
            'medape': medape_scores,
            'mape': mape_scores
        }

        results = {
            'metric': [],
            'mean': [],
            'ci_lower': [],
            'ci_upper': []
        }

        for metric, scores in metrics.items():
            mean_score = np.mean(scores)
            ci_lower = np.percentile(scores, 2.5)
            ci_upper = np.percentile(scores, 97.5)

            results['metric'].append(metric)
            results['mean'].append(mean_score)
            results['ci_lower'].append(ci_lower)
            results['ci_upper'].append(ci_upper)

        df_results = pd.DataFrame(results)

        output_dir = './bootstrap_results'
        os.makedirs(output_dir, exist_ok=True)
        df_results.to_csv(os.path.join(output_dir, 'bootstrap_metrics.csv'), index=False)
        
        return precision_scores, recall_scores, f1_scores, roc_scores, mae_scores, mse_scores, rmse_scores, medape_scores, mape_scores

    

    def plot_bootstrap(self, precision_scores, recall_scores, f1_scores, mae_scores, mse_scores):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].boxplot([precision_scores, recall_scores, f1_scores], vert=True, patch_artist=True)
        axes[0, 0].set_title('Classificação (Precision, Recall, F1)')
        axes[0, 0].set_xticklabels(['Precision', 'Recall', 'F1'])
        axes[0, 0].set_ylabel('Métricas')

        axes[0, 1].boxplot([mae_scores, mse_scores], vert=True, patch_artist=True)
        axes[0, 1].set_title('Regressão (MAE, MSE)')
        axes[0, 1].set_xticklabels(['MAE', 'MSE'])
        axes[0, 1].set_ylabel('Erro')

        all_scores = [precision_scores, recall_scores, f1_scores, mae_scores, mse_scores]
        axes[0, 2].boxplot(all_scores, vert=True, patch_artist=True)
        axes[0, 2].set_title('Distribuição Geral das Métricas')
        axes[0, 2].set_xticklabels(['Precision', 'Recall', 'F1', 'MAE', 'MSE'])
        axes[0, 2].set_ylabel('Valores')

        all_scores = [precision_scores, recall_scores, f1_scores, mae_scores, mse_scores]
        axes[0, 2].boxplot(all_scores, vert=True, patch_artist=True)
        axes[0, 2].set_title('Distribuição Geral das Métricas')
        axes[0, 2].set_xticklabels(['Precision', 'Recall', 'F1', 'MAE', 'MSE'])
        axes[0, 2].set_ylabel('Valores')

        plt.tight_layout()
        plt.show()