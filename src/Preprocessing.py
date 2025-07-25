import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Outbreak_Classification import definir_surto

def load_and_preprocess_data(filepath, window_size, numeric_columns, target_column_class, target_column_regress, timestamp_column, doenca):
    df = pd.read_parquet(filepath)
    
    df = df_transforming(df)

    df.sort_values(by='DT_NOTIFIC', ascending=True, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df = definir_surto(df, 'TAXA_INC_MES', doenca)

    df = df.drop_duplicates()

    timestamp = df[timestamp_column]

    data_scaled, scaler, y_class, y_regress, scaler_y = data_scaling(df, numeric_columns, target_column_class, target_column_regress)

    X, y_out_class, y_out_regress = create_labels(data_scaled, window_size, y_class, y_regress)

    X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress = split_data_by_time(
        X, y_out_class, y_out_regress, timestamp[window_size:]
    )

    return df, X, scaler, scaler_y, X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress, timestamp


def data_scaling(df, numeric_columns, target_column_class, target_column_regress):
    scaler = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data_num = df[numeric_columns]
    data_num_scaled = scaler.fit_transform(data_num)

    y_regress = df[target_column_regress]
    y_regress_scaled = scaler_y.fit_transform(y_regress.values.reshape(-1, 1))

    y_class = df[target_column_class].values 
    
    return data_num_scaled, scaler, y_class, y_regress_scaled, scaler_y


def create_labels(df, window_size, y_class, y_regress):
    X, y_out_class, y_out_regress = [], [], []
    
    for i in range(len(df) - window_size):
        X.append(df[i:i + window_size, :]) 
        
        y_out_class.append(y_class[i + window_size])
        
        y_out_regress.append(y_regress[i + window_size])
    
    return np.array(X), np.array(y_out_class), np.array(y_out_regress)


def split_data_by_time(df_final, y_class, y_regress, timestamp, test_start="2023-01-01", val_split=0.2, seed=2021):
    train_mask = timestamp < test_start
    test_mask = timestamp >= test_start

    X_train, X_test = df_final[train_mask], df_final[test_mask]
    y_train_class, y_test_class = y_class[train_mask], y_class[test_mask]
    y_train_regress, y_test_regress = y_regress[train_mask], y_regress[test_mask]

    return X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress

def df_transforming(df):
    df['NU_IDADE_N'] = df['NU_IDADE_N'].str.extract('(\d+)').astype(int)

    df['DIA'] = df['DT_NOTIFIC'].dt.day
    df['MES'] = df['DT_NOTIFIC'].dt.month
    df['ANO'] = df['DT_NOTIFIC'].dt.year
    df['SEMANA'] = df['DT_NOTIFIC'].dt.isocalendar().week

    populacoes = {
        'Recife': [1610000, 1625580, 1633697, 1637834, 1645727, 1653461, 1550000, 1488920, 1568854],
        'Camaragibe': [154310, 155228, 156361, 156736, 157828, 158899, 153335, 147771, 148230],
        'Cabo de Santo Agostinho': [201793, 202636, 204653, 205112, 207048, 208944, 206192, 203440, 204155],
        'Caruaru': [350028, 351686, 356128, 356872, 361118, 365278, 371663, 378048, 379573],
        'Jaboatão dos Guararapes': [689349, 691125, 695956, 697636, 702298, 706867, 675452, 644037, 645690],
        'Petrolina': [333524, 337683, 343219, 343865, 349145, 354317, 370554, 386791, 388970],
        'Olinda': [389400, 390144, 390771, 391835, 392482, 393115, 371546, 349976, 350859]
    }

    if df['ID_MUNICIP'].iloc[0] in populacoes:
        populacao_ano = pd.DataFrame({
            'ANO': list(range(2015, 2024)),
            'POPULACAO': populacoes[df['ID_MUNICIP'].iloc[0]]
        })
    else:
        return None

    df['ANO'] = df['ANO'].astype(int)
    populacao_ano['ANO'] = populacao_ano['ANO'].astype(int)

    df = pd.merge(df, populacao_ano, on='ANO', how='left')

    casos_diarios = df['DT_NOTIFIC'].value_counts().sort_index()
    df['NUM_CASOS_DIARIOS'] = df['DT_NOTIFIC'].map(casos_diarios)

    num_casos_por_semana = df.groupby(['ANO', 'SEMANA']).size().reset_index(name='NUM_CASOS_SEM')
    df = pd.merge(df, num_casos_por_semana, on=['ANO', 'SEMANA'], how='left')

    df['TAXA_INC_SEM'] = (df['NUM_CASOS_SEM'] / df['POPULACAO']) * 100000

    num_casos_por_mes = df.groupby(['ANO', 'MES']).size().reset_index(name='NUM_CASOS_MES')
    df = pd.merge(df, num_casos_por_mes, on=['ANO', 'MES'], how='left')

    df['TAXA_INC_MES'] = (df['NUM_CASOS_MES'] / df['POPULACAO']) * 100000
    
    df = df[(df['ANO'] != 2015) & (df['ANO'] != 2016) & (df['ANO'] != 2024)]

    df = agrupamento(df)

    return df

def agrupamento(df):
    agg_dict = {
        'POPULACAO': 'first',
        'NUM_CASOS_SEM': 'first',
        'TAXA_INC_SEM': 'first',
        'NUM_CASOS_MES': 'first',
        'TAXA_INC_MES': 'first',
        'DT_NOTIFIC': 'first',
        'NUM_CASOS_DIARIOS': 'first'
    }

    group_keys = ['DT_NOTIFIC']

    grouped_df = df.groupby(group_keys, as_index=False).agg(agg_dict)

    return grouped_df
