def definir_surto(df, coluna_incidencia, doenca, periodo='M'):
    df['PERIODO'] = df['DT_NOTIFIC'].dt.to_period(periodo)
    
    if doenca == 'dengue':
        coluna_incidencia  
        limiar_percentil = 0.75  
    elif doenca == 'chikungunya':
        coluna_incidencia  
        limiar_percentil = 0.70  
    elif doenca == 'zika':
        coluna_incidencia 
        limiar_percentil = 0.70  
    else:
        raise ValueError(" ")

    incidencia_periodo = df.groupby('PERIODO')[coluna_incidencia].sum()

    limiar = incidencia_periodo.quantile(limiar_percentil)
    
    def calcular_surto(periodo):
        return 1 if incidencia_periodo[periodo] > limiar else 0
    
    df['SURTO'] = df['PERIODO'].map(calcular_surto)
    
    return df