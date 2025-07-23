# plot_zika.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cases_week(filtered_df, municipe, disease):
    years = range(2016, 2025)

    for year in years:
        year_str = str(year)

        filtered_data = filtered_df[
            (filtered_df['ID_MUNICIP'] == municipe) & 
            (filtered_df['NU_ANO'] == year_str)
        ]

        weekly_counts = filtered_data.groupby('SEM_NOT').size().reset_index(name='Quantidade de Casos')
        weekly_counts = weekly_counts.sort_values(by='SEM_NOT')

        plt.figure(figsize=(14, 8))
        sns.set_palette('pastel')

        barplot = sns.barplot(data=weekly_counts, x='SEM_NOT', y='Quantidade de Casos', color='steelblue')

        for container in barplot.containers:
            barplot.bar_label(container, label_type='edge', fontsize=8, fontweight='bold')

        for bar in barplot.patches:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, ha='right')

        plt.title(f'Quantidade de Casos de {disease} por Semana em {municipe} - {year}')
        plt.xlabel('Semana')
        plt.ylabel('Quantidade de Casos')
        plt.tight_layout()

        plt.show()


def plot_cases_year(filtered_df, disease):

    count_df = filtered_df.groupby(['ID_MUNICIP', 'NU_ANO']).size().reset_index(name='Quantidade de Casos')

    count_df['NU_ANO'] = pd.to_numeric(count_df['NU_ANO'], errors='coerce')
    count_df = count_df.dropna(subset=['NU_ANO'])
    count_df['NU_ANO'] = count_df['NU_ANO'].astype(int)
    count_df = count_df.sort_values(by='NU_ANO')

    plt.figure(figsize=(14, 8))
    sns.set_palette('pastel')

    barplot = sns.barplot(data=count_df, x='NU_ANO', y='Quantidade de Casos', hue='ID_MUNICIP')

    for container in barplot.containers:
        barplot.bar_label(container, label_type='edge', fontsize=8, fontweight='bold')

    for bar in barplot.patches:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)

    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, ha='right')

    plt.title(f'Quantidade de Casos de {disease} por Município')
    plt.xlabel('Ano')
    plt.ylabel('Quantidade de Casos')
    plt.tight_layout()

    plt.show()


def plot_train_test(timestamp, y):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(timestamp[timestamp < '2023-01-01'], y[timestamp < '2023-01-01'], color='black', lw=2)
    plt.plot(timestamp[timestamp >= '2023-01-01'], y[timestamp >= '2023-01-01'], color='#6593a4', lw=2)
    plt.title('Divisão de Dados de Treinamento e Teste', fontsize=15)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Número de Casos de Dengue (Escalonado)', fontsize=12)
    plt.legend(['Conjunto de Treinamento', 'Conjunto de Teste'], loc='upper left', prop={'size': 15})
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()



def plot_loss(histories, model_name):
    plt.figure(figsize=(12, 5))
    plt.plot(histories[0].history['loss'], label='Loss (Train)')
    plt.plot(histories[0].history['val_loss'], label='Loss (Validation)')
    plt.title(f'Loss durante o treinamento do {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid()
    plt.show()


def outbreak_plot(df, disease):
    plt.figure(figsize=(10, 6))
    plt.plot(df['DT_NOTIFIC'], df['TAXA_INC_MES'], color='black', markersize=4)
    plt.scatter(df[df['SURTO'] == 1]['DT_NOTIFIC'], df[df['SURTO'] == 1]['TAXA_INC_MES'], color='#048ABF')
    plt.title(f'Monthly Incidence Rate in Recife ({disease})', fontsize=14)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()