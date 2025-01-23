import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

# --- Funções ---
def CriarGraficos(fig, graph_num):
    # Cria o diretório 'images' se ele não existir
    if not os.path.exists("images"):
        os.makedirs("images")

    # Nomes dos arquivos de saída
    output_file_jpeg = f"images/grafico_{graph_num}_high_quality.jpg"
    output_file_png = f"images/grafico_{graph_num}_high_quality.png"
    output_file_svg = f"images/grafico_{graph_num}_high_quality.svg"

    # Salvando as imagens nos diferentes formatos
    fig.write_image(output_file_jpeg, format='jpeg', scale=3)
    fig.write_image(output_file_png, format='png', scale=3)
    fig.write_image(output_file_svg, format='svg', scale=3)

    # Criando colunas para os botões de download
    left, middle, right = st.columns(3)

    # Botão de download para JPEG
    with open(output_file_jpeg, "rb") as file:
        left.download_button(
            label=f"Baixar Gráfico {graph_num} em JPEG",
            data=file,
            file_name=output_file_jpeg,
            mime="image/jpeg",
            key=f"download_jpeg_{graph_num}",
        )

    # Botão de download para PNG
    with open(output_file_png, "rb") as file:
        middle.download_button(
            label=f"Baixar Gráfico {graph_num} em PNG",
            data=file,
            file_name=output_file_png,
            mime="image/png",
            key=f"download_png_{graph_num}",
        )

    # Botão de download para SVG
    with open(output_file_svg, "rb") as file:
        right.download_button(
            label=f"Baixar Gráfico {graph_num} em SVG",
            data=file,
            file_name=output_file_svg,
            mime="image/svg+xml",
            key=f"download_svg_{graph_num}",
        )


def executar_previsao_carga(dados, learning_rate, num_neurons, activation, passos_anteriores):
    # Carregando os dados
    carga = dados.iloc[10:1010, 14].values
    scaler = StandardScaler()
    carga_normalizada = scaler.fit_transform(carga.reshape(-1, 1)).flatten()

    # Criar dados em formato de série temporal
    def criar_dados(series, passos=passos_anteriores):
        X, Y = [], []
        for i in range(len(series) - passos):
            X.append(series[i:i + passos])
            Y.append(series[i + passos])
        return np.array(X), np.array(Y)

    # Preparar os dados
    X, Y = criar_dados(carga_normalizada, passos_anteriores)

    # Dividir os dados em treino e teste
    split = int(0.6 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    # Ajustar o formato para LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Criar um Dataset para treinar eficientemente o modelo LSTM usando o TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(st.session_state['batch_size']).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.batch(st.session_state['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Criar o modelo LSTM com regularização L2
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(passos_anteriores, 1)),
        tf.keras.layers.LSTM(
            num_neurons,
            activation=activation,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.LSTM(num_neurons, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    # model.summary() # Descomente para exibir o resumo do modelo

    # Configurar callbacks para o treinamento
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    )

    # Treinar o modelo
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=st.session_state['epochs'],
        batch_size=st.session_state['batch_size'],
        shuffle=False,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    # Fazer previsões
    previsoes = model.predict(X_test, batch_size=st.session_state['batch_size'])
    previsoes = scaler.inverse_transform(previsoes)
    Y_test_real = scaler.inverse_transform(Y_test.reshape(-1, 1))
    carga_prevista = previsoes

    # Avaliar o modelo
    MAPE = np.mean(np.abs((previsoes - Y_test_real) / Y_test_real)) * 100
    MAE = np.mean(np.abs(previsoes - Y_test_real))
    RMSE = np.sqrt(np.mean((previsoes - Y_test_real) ** 2))

    # Exibir métricas
    st.subheader('Métricas de Desempenho da Previsão')
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {MAPE:.4f}%")
    st.write(f"**Erro Absoluto Médio (MAE):** {MAE:.4f}")
    st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** {RMSE:.4f}")

    timesteps = list(range(1, len(Y_test_real) + 1))

    # Criar o gráfico interativo
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=Y_test_real.flatten(), mode='lines', name='Dados Reais', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=previsoes.flatten(), mode='lines', name='Previsão', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Previsão de Carga usando LSTM', xaxis_title='Tempo', yaxis_title='Carga',
                      legend=dict(x=0, y=1), template='plotly_white')

    st.plotly_chart(fig)
    CriarGraficos(fig, 0)

    return carga_prevista, timesteps

def executar_despacho_economico(dados, carga_prevista, preco_solar, preco_eolica, preco_h2, preco_bateria, dolar_hoje, Potenciadosistema):
    # Leitura dos dados
    geracao = dados

    # Parâmetros iniciais
    G_Solar = np.zeros(288)
    Radiacao = np.array(geracao.iloc[18:306, 7])
    Temperatura = np.array(geracao.iloc[18:306, 6])
    Vento = np.array(geracao.iloc[18:306, 4])

    # Cálculo da geração solar
    for i in range(len(Radiacao)):
        G_Solar[i] = 0.97 * (Radiacao[i]) / 1000 * (1 + 0.005 * ((Temperatura[i]) - 25))

    # Parâmetros eólicos
    n_WT = 1
    eta_WT = 0.9
    P_R_WT = 1
    u_cut_in = 3
    u_rated = 12
    u_cut_off = 25

    # Potência gerada pelo vento
    P_WT = np.zeros(len(Vento))
    for i, u in enumerate(Vento):
        if u < u_cut_in:
            P_WT[i] = 0
        elif u_cut_in <= u <= u_rated:
            P_WT[i] = n_WT * eta_WT * P_R_WT * ((u ** 2 - u_cut_in ** 2) / (u_rated ** 2 - u_cut_in ** 2))
        elif u_rated < u < u_cut_off:
            P_WT[i] = n_WT * eta_WT * P_R_WT
        else:
            P_WT[i] = 0

    # Parâmetros gerais
    potenciadosistema = Potenciadosistema
    potenciaEolicamaxima_values = P_WT * potenciadosistema
    potenciasolarmaxima_values = G_Solar * potenciadosistema
    Carga_Residencial = np.array(carga_prevista[18:306])

    horas_uso_cel_comb = 8760
    vida_util_sistema = 20
    precoH2 = preco_h2
    precoeolica = preco_eolica
    precosolar = preco_solar
    precobateria = preco_bateria
    tanque_hidrogenio_max = 45
    nivel_tanque = tanque_hidrogenio_max
    consumo_kg_h2 = 0.2
    eficiencia_eletrolisador = 0.75
    consumo_kW_h2 = -53571 * eficiencia_eletrolisador + 92643
    potencia_maxima_eletrolisador = Potenciadosistema
    niveis_eletrolisador = 6
    cel_combustivel = 20000

    # Possibilidades do eletrólise
    possibilidades_eletrolisador = [0] + [i * potencia_maxima_eletrolisador / niveis_eletrolisador for i in range(1, niveis_eletrolisador + 1)]

    # Inicialização
    pbat = 30000
    potenciaBateria_max = 27000
    potenciaH2maxima = 30000
    status_bateria = np.zeros(len(Carga_Residencial))
    status_tanque = np.zeros(len(Carga_Residencial))

    # Inicializando listas para armazenar os resultados
    PotenciaSolar = []
    PotenciaEolica = []
    CustoTotal = []
    PotenciaH2 = []
    PotenciaBateria = []

    for i in range(len(potenciasolarmaxima_values)):
        status_bateria[i] = potenciaBateria_max
        status_tanque[i] = nivel_tanque
        potenciasolarmaxima = potenciasolarmaxima_values[i]
        potenciaEolicamaxima = potenciaEolicamaxima_values[i]

        preco_eolica_calc = dolar_hoje * ((potenciaEolicamaxima / 1000) * (1000 + 30) * precoeolica + vida_util_sistema * 20)
        preco_solar_calc = dolar_hoje * ((potenciasolarmaxima / 1000) * (2000 + 300 + 10) * precosolar + vida_util_sistema * (50 + 10))
        preco_bateria_calc = dolar_hoje * ((potenciaBateria_max * 200 / 1000) * (precobateria + 0.19) + vida_util_sistema * 10)
        preco_cel_comb = dolar_hoje * ((cel_combustivel * 3000 / 1000) + vida_util_sistema * 0.02 * horas_uso_cel_comb)
        preco_eletrolisador = dolar_hoje * ((potencia_maxima_eletrolisador * 500 / 1000) * precoH2 + vida_util_sistema * 10)
        preco_tanque = dolar_hoje * nivel_tanque * (500 + vida_util_sistema * 10)
        PrecoH2_calc = (preco_cel_comb + preco_eletrolisador + preco_tanque)

        potencia_disponivel = potenciasolarmaxima + potenciaEolicamaxima
        limite_minimo_bateria = 0.1 * pbat

        if Carga_Residencial[i][0] <= potencia_disponivel:
            c = [preco_solar_calc, preco_eolica_calc, 0, 0]
            A_eq = [[1, 1, 0, 0]]
            b_eq = [Carga_Residencial[i][0]]
            bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, 0), (0, 0)]
        elif Carga_Residencial[i][0] <= potencia_disponivel + potenciaBateria_max and potenciaBateria_max > limite_minimo_bateria:
            c = [preco_solar_calc, preco_eolica_calc, 0, preco_bateria_calc]
            A_eq = [[1, 1, 0, 1]]
            b_eq = [Carga_Residencial[i][0]]
            bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, 0), (limite_minimo_bateria + 1, potenciaBateria_max)]
        else:
            c = [preco_solar_calc, preco_eolica_calc, PrecoH2_calc, 0]
            A_eq = [[1, 1, 1, 0]]
            b_eq = [Carga_Residencial[i][0]]
            bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, potenciaH2maxima), (0, 0)]

        c = np.array(c).flatten()
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        PotenciaBateria.append(result.x[3])
        PotenciaSolar.append(result.x[0])
        PotenciaEolica.append(result.x[1])
        CustoTotal.append(result.fun)
        PotenciaH2.append(result.x[2])

        possivel_carga_bateria = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][0]
        possivel_carga_H2 = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][0] - possivel_carga_bateria

        if possivel_carga_bateria < 0:
            possivel_carga_bateria = 0
        if possivel_carga_H2 < 0:
            possivel_carga_H2 = 0
        if possivel_carga_bateria > pbat - potenciaBateria_max:
            possivel_carga_bateria = pbat - potenciaBateria_max

        if nivel_tanque >= tanque_hidrogenio_max:
            possivel_carga_H2 = 0
        else:
            for n in range(2, niveis_eletrolisador + 2):
                if possivel_carga_H2 < possibilidades_eletrolisador[n - 1]:
                    possivel_carga_H2 = possibilidades_eletrolisador[n - 2]
                if n == (niveis_eletrolisador + 1) and possivel_carga_H2 >= possibilidades_eletrolisador[n - 1]:
                    possivel_carga_H2 = possibilidades_eletrolisador[n - 1]
        if tanque_hidrogenio_max - nivel_tanque < possivel_carga_H2 / consumo_kW_h2:
            possivel_carga_H2 = ((tanque_hidrogenio_max - nivel_tanque) * consumo_kW_h2)

        if result.success:
            x = result.x
            potenciaBateria_max = potenciaBateria_max - x[3] + possivel_carga_bateria
            nivel_tanque = (nivel_tanque - (x[2] / 1000) * consumo_kg_h2 + (possivel_carga_H2 / consumo_kW_h2))
        if potenciaBateria_max < 3000:
            potenciaBateria_max = 3000

    PotenciaTotal = []
    for i in range(len(PotenciaEolica)):
        PotenciaTotal.append(potenciasolarmaxima_values[i] + potenciaEolicamaxima_values[i])

    Custo_dia = sum(CustoTotal) / 1000 / (288)
    status_bateria = status_bateria * 100 / pbat
    status_tanque = status_tanque * 100 / tanque_hidrogenio_max

    st.subheader("Resultados do Despacho Econômico")
    st.success(
        "Otimização concluída, custo total para implementação e operação durante um ano: R$ {:,.2f}".format(Custo_dia))
    st.markdown(f"Vida útil do sistema: {vida_util_sistema:.2f} anos")

    return PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, status_bateria, status_tanque

    def executar_despacho_economico(dados, carga_prevista, preco_solar, preco_eolica, preco_h2, preco_bateria,
                                    dolar_hoje, Potenciadosistema):
        # Leitura dos dados
        geracao = dados

        # Parâmetros iniciais
        G_Solar = np.zeros(288)
        Radiacao = np.array(geracao.iloc[18:306, 7])
        Temperatura = np.array(geracao.iloc[18:306, 6])
        Vento = np.array(geracao.iloc[18:306, 4])

        # Cálculo da geração solar
        for i in range(len(Radiacao)):
            G_Solar[i] = 0.97 * (Radiacao[i]) / 1000 * (1 + 0.005 * ((Temperatura[i]) - 25))

        # Parâmetros eólicos
        n_WT = 1
        eta_WT = 0.9
        P_R_WT = 1
        u_cut_in = 3
        u_rated = 12
        u_cut_off = 25

        # Potência gerada pelo vento
        P_WT = np.zeros(len(Vento))
        for i, u in enumerate(Vento):
            if u < u_cut_in:
                P_WT[i] = 0
            elif u_cut_in <= u <= u_rated:
                P_WT[i] = n_WT * eta_WT * P_R_WT * ((u ** 2 - u_cut_in ** 2) / (u_rated ** 2 - u_cut_in ** 2))
            elif u_rated < u < u_cut_off:
                P_WT[i] = n_WT * eta_WT * P_R_WT
            else:
                P_WT[i] = 0

        # Parâmetros gerais
        potenciadosistema = Potenciadosistema
        potenciaEolicamaxima_values = P_WT * potenciadosistema
        potenciasolarmaxima_values = G_Solar * potenciadosistema
        Carga_Residencial = np.array(carga_prevista[18:306])

        horas_uso_cel_comb = 8760
        vida_util_sistema = 20
        precoH2 = preco_h2
        precoeolica = preco_eolica
        precosolar = preco_solar
        precobateria = preco_bateria
        tanque_hidrogenio_max = 45
        nivel_tanque = tanque_hidrogenio_max
        consumo_kg_h2 = 0.2
        eficiencia_eletrolisador = 0.75
        consumo_kW_h2 = -53571 * eficiencia_eletrolisador + 92643
        potencia_maxima_eletrolisador = Potenciadosistema
        niveis_eletrolisador = 6
        cel_combustivel = 20000

        # Possibilidades do eletrólise
        possibilidades_eletrolisador = [0] + [i * potencia_maxima_eletrolisador / niveis_eletrolisador for i in
                                              range(1, niveis_eletrolisador + 1)]

        # Inicialização
        pbat = 30000
        potenciaBateria_max = 27000
        potenciaH2maxima = 30000
        status_bateria = np.zeros(len(Carga_Residencial))
        status_tanque = np.zeros(len(Carga_Residencial))

        # Inicializando listas para armazenar os resultados
        PotenciaSolar = []
        PotenciaEolica = []
        CustoTotal = []
        PotenciaH2 = []
        PotenciaBateria = []

        for i in range(len(potenciasolarmaxima_values)):
            status_bateria[i] = potenciaBateria_max
            status_tanque[i] = nivel_tanque
            potenciasolarmaxima = potenciasolarmaxima_values[i]
            potenciaEolicamaxima = potenciaEolicamaxima_values[i]

            preco_eolica_calc = dolar_hoje * (
                        (potenciaEolicamaxima / 1000) * (1000 + 30) * precoeolica + vida_util_sistema * 20)
            preco_solar_calc = dolar_hoje * (
                        (potenciasolarmaxima / 1000) * (2000 + 300 + 10) * precosolar + vida_util_sistema * (50 + 10))
            preco_bateria_calc = dolar_hoje * (
                        (potenciaBateria_max * 200 / 1000) * (precobateria + 0.19) + vida_util_sistema * 10)
            preco_cel_comb = dolar_hoje * (
                        (cel_combustivel * 3000 / 1000) + vida_util_sistema * 0.02 * horas_uso_cel_comb)
            preco_eletrolisador = dolar_hoje * (
                        (potencia_maxima_eletrolisador * 500 / 1000) * precoH2 + vida_util_sistema * 10)
            preco_tanque = dolar_hoje * nivel_tanque * (500 + vida_util_sistema * 10)
            PrecoH2_calc = (preco_cel_comb + preco_eletrolisador + preco_tanque)

            potencia_disponivel = potenciasolarmaxima + potenciaEolicamaxima
            limite_minimo_bateria = 0.1 * pbat

            if Carga_Residencial[i][0] <= potencia_disponivel:
                c = [preco_solar_calc, preco_eolica_calc, 0, 0]
                A_eq = [[1, 1, 0, 0]]
                b_eq = [Carga_Residencial[i][0]]
                bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, 0), (0, 0)]
            elif Carga_Residencial[i][
                0] <= potencia_disponivel + potenciaBateria_max and potenciaBateria_max > limite_minimo_bateria:
                c = [preco_solar_calc, preco_eolica_calc, 0, preco_bateria_calc]
                A_eq = [[1, 1, 0, 1]]
                b_eq = [Carga_Residencial[i][0]]
                bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, 0),
                          (limite_minimo_bateria + 1, potenciaBateria_max)]
            else:
                c = [preco_solar_calc, preco_eolica_calc, PrecoH2_calc, 0]
                A_eq = [[1, 1, 1, 0]]
                b_eq = [Carga_Residencial[i][0]]
                bounds = [(0, potenciasolarmaxima), (0, potenciaEolicamaxima), (0, potenciaH2maxima), (0, 0)]

            c = np.array(c).flatten()
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            PotenciaBateria.append(result.x[3])
            PotenciaSolar.append(result.x[0])
            PotenciaEolica.append(result.x[1])
            CustoTotal.append(result.fun)
            PotenciaH2.append(result.x[2])

            possivel_carga_bateria = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][0]
            possivel_carga_H2 = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][
                0] - possivel_carga_bateria

            if possivel_carga_bateria < 0:
                possivel_carga_bateria = 0
            if possivel_carga_H2 < 0:
                possivel_carga_H2 = 0
            if possivel_carga_bateria > pbat - potenciaBateria_max:
                possivel_carga_bateria = pbat - potenciaBateria_max

            if nivel_tanque >= tanque_hidrogenio_max:
                possivel_carga_H2 = 0
            else:
                for n in range(2, niveis_eletrolisador + 2):
                    if possivel_carga_H2 < possibilidades_eletrolisador[n - 1]:
                        possivel_carga_H2 = possibilidades_eletrolisador[n - 2]
                    if n == (niveis_eletrolisador + 1) and possivel_carga_H2 >= possibilidades_eletrolisador[n - 1]:
                        possivel_carga_H2 = possibilidades_eletrolisador[n - 1]
            if tanque_hidrogenio_max - nivel_tanque < possivel_carga_H2 / consumo_kW_h2:
                possivel_carga_H2 = ((tanque_hidrogenio_max - nivel_tanque) * consumo_kW_h2)

            if result.success:
                x = result.x
                potenciaBateria_max = potenciaBateria_max - x[3] + possivel_carga_bateria
                nivel_tanque = (nivel_tanque - (x[2] / 1000) * consumo_kg_h2 + (possivel_carga_H2 / consumo_kW_h2))
            if potenciaBateria_max < 3000:
                potenciaBateria_max = 3000

            PotenciaTotal = []
            for i in range(len(PotenciaEolica)):
                PotenciaTotal.append(potenciasolarmaxima_values[i] + potenciaEolicamaxima_values[i])

            Custo_dia = sum(CustoTotal) / 1000 / (288)
            status_bateria = status_bateria * 100 / pbat
            status_tanque = status_tanque * 100 / tanque_hidrogenio_max

            st.subheader("Resultados do Despacho Econômico")
            st.success(
                "Otimização concluída, custo total para implementação e operação durante um ano: R$ {:,.2f}".format(
                    Custo_dia))
            st.markdown(f"Vida útil do sistema: {vida_util_sistema:.2f} anos")

            return PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, status_bateria, status_tanque

        def FazAsImagens(timesteps, PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal,
                         PotenciaTotal, previsoes, status_bateria, status_tanque):
            # Gráfico 1: Potências ao longo do tempo
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(x=timesteps, y=PotenciaSolar, mode='lines+markers', name='Potência Solar',
                           line=dict(color='yellow'),
                           marker=dict(color='yellow')))
            fig1.add_trace(
                go.Scatter(x=timesteps, y=PotenciaEolica, mode='lines+markers', name='Potência Eólica',
                           line=dict(color='blue'),
                           marker=dict(color='blue')))
            fig1.add_trace(
                go.Scatter(x=timesteps, y=PotenciaH2, mode='lines+markers', name='Potência H2',
                           line=dict(color='green'),
                           marker=dict(color='green')))
            fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaBateria, mode='lines+markers', name='Potência Bateria',
                                      line=dict(color='purple'), marker=dict(color='purple')))
            fig1.update_layout(title="Potências ao Longo do Tempo", xaxis_title="Etapas de Tempo",
                               yaxis_title="Potência (kW)",
                               width=1200, height=450, template="plotly_white")
            st.plotly_chart(fig1)
            CriarGraficos(fig1, 1)

            # Gráfico 2: Custo Total
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=timesteps, y=CustoTotal, mode='lines+markers', name='Custo Total'))
            fig2.update_layout(title="Custo Total ao Longo do Tempo", xaxis_title="Etapas de Tempo",
                               yaxis_title="Custo ($)", template="plotly_white")
            st.plotly_chart(fig2)
            CriarGraficos(fig2, 2)

            # Gráfico 3: Potência Total e Carga Residencial
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(x=timesteps, y=PotenciaTotal, mode='lines+markers', name='Potência Total',
                           line=dict(color='blue')))
            fig3.add_trace(
                go.Scatter(x=timesteps, y=previsoes.flatten(), mode='lines+markers', name='Carga Residencial',
                           line=dict(color='green')))
            fig3.update_layout(title="Potência Total vs Carga Residencial", xaxis_title="Etapas de Tempo",
                               yaxis_title="Potência (kW)", template="plotly_white")
            st.plotly_chart(fig3)
            CriarGraficos(fig3, 3)

            # Gráfico 4: Status da Bateria e Tanque
            fig4 = go.Figure()
            fig4.add_trace(
                go.Scatter(x=timesteps, y=status_bateria, mode='lines+markers', name='Bateria (%)',
                           line=dict(color='purple'),
                           marker=dict(color='purple')))
            fig4.add_trace(go.Scatter(x=timesteps, y=status_tanque, mode='lines+markers', name='Tanque de H2 (%)',
                                      line=dict(color='green'), marker=dict(color='green')))
            fig4.update_layout(title="Status da Bateria e Tanque de H2", xaxis_title="Etapas de Tempo",
                               yaxis_title="Status (%)", template="plotly_white")
            st.plotly_chart(fig4)
            CriarGraficos(fig4, 4)
            st.success("Despacho concluído!")
# --- Interface do Streamlit ---
st.title("Previsão de Carga e Despacho Econômico")

# Parâmetros de Previsão de Carga
st.sidebar.subheader("Parâmetros de Previsão de Carga")

if 'epochs' not in st.session_state:
    st.session_state['epochs'] = 50
if 'batch_size' not in st.session_state:
    st.session_state['batch_size'] = 32
if 'learning_rate' not in st.session_state:
    st.session_state['learning_rate'] = 0.001
if 'num_neurons' not in st.session_state:
    st.session_state['num_neurons'] = 64
if 'activation' not in st.session_state:
    st.session_state['activation'] = 'relu'
if 'passos_anteriores' not in st.session_state:
    st.session_state['passos_anteriores'] = 10

st.session_state['epochs'] = st.sidebar.number_input("Quantas epocas simular", value=st.session_state['epochs'])
st.session_state['batch_size'] = st.sidebar.number_input("Tamanho do lote",
                                                         value=st.session_state['batch_size'])
st.session_state['learning_rate'] = st.sidebar.number_input("Taxa de aprendizado",
                                                            value=st.session_state['learning_rate'],
                                                            format="%.3f")
st.session_state['num_neurons'] = st.sidebar.number_input("Número de neurônios por camada",
                                                          value=st.session_state['num_neurons'])
st.session_state['activation'] = st.sidebar.selectbox("Função de ativação", ["relu", "sigmoid", "tanh"],
                                                      index=['relu', 'sigmoid', 'tanh'].index(
                                                          st.session_state['activation']))
st.session_state['passos_anteriores'] = st.sidebar.number_input("Passos anteriores para previsão",
                                                                value=st.session_state['passos_anteriores'])

# Parâmetros de Despacho Econômico
st.sidebar.subheader("Parâmetros do Despacho Econômico")

if 'preco_solar' not in st.session_state:
    st.session_state['preco_solar'] = 0.2
if 'preco_eolica' not in st.session_state:
    st.session_state['preco_eolica'] = 0.2
if 'preco_h2' not in st.session_state:
    st.session_state['preco_h2'] = 14.3
if 'preco_bateria' not in st.session_state:
    st.session_state['preco_bateria'] = 0.1
if 'dolar_hoje' not in st.session_state:
    st.session_state['dolar_hoje'] = 5.1
if 'Potenciadosistema' not in st.session_state:
    st.session_state['Potenciadosistema'] = 30000

st.session_state['preco_solar'] = st.sidebar.number_input("Preço Solar ($/kW)",
                                                          value=st.session_state['preco_solar'])
st.session_state['preco_eolica'] = st.sidebar.number_input("Preço Eólica ($/kW)",
                                                           value=st.session_state['preco_eolica'])
st.session_state['preco_h2'] = st.sidebar.number_input("Preço H2 ($/kW)", value=st.session_state['preco_h2'])
st.session_state['preco_bateria'] = st.sidebar.number_input("Preço Bateria ($/kW)",
                                                            value=st.session_state['preco_bateria'])
st.session_state['dolar_hoje'] = st.sidebar.number_input("Dólar Hoje", value=st.session_state['dolar_hoje'])
st.session_state['Potenciadosistema'] = st.sidebar.number_input("Potência do Sistema (kW)",
                                                                value=st.session_state['Potenciadosistema'])

# Upload do arquivo
def carregar_dados():
    if "dados_carga" not in st.session_state:
        st.session_state["dados_carga"] = None

    uploaded_file = st.file_uploader("Carregue o arquivo de dados da carga (Excel)", type="xlsx")
    if uploaded_file is not None:
        st.session_state["dados_carga"] = pd.read_excel(uploaded_file)
        st.success("Arquivo carregado com sucesso!")

carregar_dados()

# Execução da Previsão e Despacho
if st.session_state["dados_carga"] is not None:
    if st.sidebar.button("Executar Previsão e Despacho"):
        # Limpa os dados antigos se houver
        st.session_state.pop('carga_prevista', None)
        st.session_state.pop('timesteps', None)

        # Executa a previsão
        carga_prevista, timesteps = executar_previsao_carga(
            st.session_state["dados_carga"],
            st.session_state['learning_rate'],
            st.session_state['num_neurons'],
            st.session_state['activation'],
            st.session_state['passos_anteriores']
        )

        st.session_state['carga_prevista'] = carga_prevista
        st.session_state['timesteps'] = timesteps

        # Executa o despacho
        PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, status_bateria, status_tanque = executar_despacho_economico(
            st.session_state["dados_carga"],
            st.session_state['carga_prevista'],
            st.session_state['preco_solar'],
            st.session_state['preco_eolica'],
            st.session_state['preco_h2'],
            st.session_state['preco_bateria'],
            st.session_state['dolar_hoje'],
            st.session_state['Potenciadosistema']
        )

        st.session_state['PotenciaSolar'] = PotenciaSolar
        st.session_state['PotenciaEolica'] = PotenciaEolica
        st.session_state['PotenciaH2'] = PotenciaH2
        st.session_state['PotenciaBateria'] = PotenciaBateria
        st.session_state['CustoTotal'] = CustoTotal
        st.session_state['PotenciaTotal'] = PotenciaTotal
        st.session_state['status_bateria'] = status_bateria
        st.session_state['status_tanque'] = status_tanque

        # Cria os gráficos
        FazAsImagens(
            st.session_state['timesteps'],
            st.session_state['PotenciaSolar'],
            st.session_state['PotenciaEolica'],
            st.session_state['PotenciaH2'],
            st.session_state['PotenciaBateria'],
            st.session_state['CustoTotal'],
            st.session_state['PotenciaTotal'],
            st.session_state['carga_prevista'],
            st.session_state['status_bateria'],
            st.session_state['status_tanque']
        )
else:
    st.warning("Por favor, carregue um arquivo Excel para continuar.")