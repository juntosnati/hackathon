import os
import pandas as pd
from classification import image_classification
from frame_capture import frame_capture

# Faz uma classificação bayesiana de todos os gestos computados
# para dizer qual é a frase dita
def sign_classification(classification_list):
    signs = pd.read_csv('./dataset/gestos.csv', encoding='ISO-8859-1')
    # Todos os gestos computados até o momento
    sign_list = {'ajuda_gesto1': 0, 'ajuda_gesto2': 0, 'alergia_gesto1': 0, 'alergia_gesto2': 0, 'banheiro_gesto1': 0, 
                        'banheiro_gesto2': 0, 'bom_dia_gesto1': 0, 'bom_dia_gesto2': 0, 'bom_dia_gesto3': 0, 'desculpa_gesto1': 0, 
                        'enjoo_gesto1': 0, 'enjoo_gesto2': 0, 'estou_mal_gesto1': 0, 'estou_mal_gesto2': 0, 'frase9_gesto1': 0, 
                        'frase9_gesto2': 0, 'meu_nome_e_gesto1': 0, 'meu_nome_e_gesto2': 0, 'por_favor_gesto1': 0}
    from sklearn.naive_bayes import MultinomialNB
    X = signs[signs.columns[1:]]
    # O gesto 'Parada' ainda não foi implementado
    X = X.drop(['parada'], axis=1)
    y = signs['frase']
    model = MultinomialNB()
    model.fit(X, y)
    # Percorre todos os gestos recebidos pelo pela função chama_classificacao
    # e atualiza o dicionário 'lista_de_gestos' com o valor 1 em todo gesto que aparecer
    for sign in classification_list:
        sign_list.update({sign: 1})

    cl_list = model.predict([list(sign_list.values())])
    return cl_list