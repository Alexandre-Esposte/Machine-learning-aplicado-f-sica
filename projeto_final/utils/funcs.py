import pandas as pd

def separa_pontos_manual(intensidade_relativa = 0.1,centro = 0,linha = 0, spec_mensurado = None):

    #global x_espec,y_espec

    #linha = lines[id].tolist()

    #centro = float(linha[2])

    x_espec = list(spec_mensurado['wavenumber'])
    y_espec = list(spec_mensurado['intensity'])
    

    pontosx = []
    pontosy = []

    # intensidade do centro e dos pontos adjacentes
    icentro = float(linha)
    iponto = float(linha)# só inicializando a variável, preciso disso para entrar no while (1 > intensidade_relativa)

    esquerda_x = []
    esquerda_y = []

    direita_x = []
    direita_y = []


    i = 1

    # para a esquerda
    while (iponto / icentro >= intensidade_relativa):


        # aqui é de fato o valor da intensidade dos pontos adjacentes
        iponto = y_espec[y_espec.index(icentro) - i]



        if iponto / icentro >= intensidade_relativa:
            esquerda_x.insert(-i, x_espec[x_espec.index(centro) - i])
            esquerda_y.insert(-i, y_espec[y_espec.index(icentro) - i])


        i = i + 1

    # Para direita

    icentro = float(linha)
    iponto  = float(linha)
    i = 1
    while (iponto / icentro >= intensidade_relativa):

        iponto = y_espec[y_espec.index(icentro) + i]


        if iponto / icentro >= intensidade_relativa:
            direita_x.insert(i, x_espec[x_espec.index(centro) + i])
            direita_y.insert(i, y_espec[y_espec.index(icentro) + i])


        i = i + 1

    pontos_x = esquerda_x + [centro] + direita_x

    pontos_y = esquerda_y + [icentro] + direita_y

   

    return pd.DataFrame({'wavenumber':pontos_x, 'intensity':pontos_y})