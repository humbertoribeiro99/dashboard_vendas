import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Título ---
st.title('Dashboard de Análise de Vendas')

# --- Carregar dados ---
@st.cache_data

def carregar_dados(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['data'])
    else:
        df = pd.read_csv('data/vendas.csv', parse_dates=['data'])
    df['produto'] = df['produto'].str.strip()
    df['regiao'] = df['regiao'].str.strip()
    df['total_venda'] = df['quantidade'] * df['preco_unitario']
    return df

uploaded_file = st.sidebar.file_uploader("📁 Faça upload de um novo arquivo CSV", type=["csv"])
df = carregar_dados(uploaded_file)

# --- Mostrar dados ---
if st.checkbox('Mostrar dados brutos'):
    st.write(df)

# --- Métricas gerais ---
st.header('Métricas Gerais')
total_vendas = df['total_venda'].sum()
total_quantidade = df['quantidade'].sum()
media_preco = df['preco_unitario'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total em Vendas", f"R$ {total_vendas:,.2f}")
col2.metric("Quantidade Vendida", total_quantidade)
col3.metric("Preço Médio", f"R$ {media_preco:.2f}")

# --- Análise temporal ---
st.header('Análise Temporal')
vendas_por_data = df.groupby('data')['total_venda'].sum()

fig, ax = plt.subplots()
sns.lineplot(x=vendas_por_data.index, y=vendas_por_data.values, ax=ax)
ax.set_title('Vendas por Data')
ax.set_ylabel('Total em Vendas')
ax.set_xlabel('Data')
st.pyplot(fig)

# --- Vendas por Região ---
st.header('Vendas por Região')
vendas_por_regiao = df.groupby('regiao')['total_venda'].sum().sort_values(ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(x=vendas_por_regiao.index, y=vendas_por_regiao.values, ax=ax2)
ax2.set_title('Vendas por Região')
ax2.set_ylabel('Total em Vendas')
ax2.set_xlabel('Região')
st.pyplot(fig2)

# --- Vendas por Produto ---
st.header('Vendas por Produto')
vendas_por_produto = df.groupby('produto')['total_venda'].sum().sort_values(ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(x=vendas_por_produto.index, y=vendas_por_produto.values, ax=ax3)
ax3.set_title('Vendas por Produto')
ax3.set_ylabel('Total em Vendas')
ax3.set_xlabel('Produto')
st.pyplot(fig3)

# --- Filtros ---
st.header('Filtrar dados')

produtos = df['produto'].unique()
regioes = df['regiao'].unique()

filtro_produto = st.multiselect('Produto', produtos, default=produtos)
filtro_regiao = st.multiselect('Região', regioes, default=regioes)

# --- Período ---
min_date = df['data'].min()
max_date = df['data'].max()
periodo = st.date_input("Período", [min_date, max_date])

# --- Aplicar filtros ---
df_filtrado = df[(df['produto'].isin(filtro_produto)) &
                 (df['regiao'].isin(filtro_regiao)) &
                 (df['data'] >= pd.to_datetime(periodo[0])) &
                 (df['data'] <= pd.to_datetime(periodo[1]))]

st.write(f'Dados filtrados: {len(df_filtrado)} registros')

# --- Exportar dados filtrados ---
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Vendas')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

excel = to_excel(df_filtrado)
st.download_button(label="📥 Baixar dados filtrados em Excel", data=excel, file_name='dados_filtrados.xlsx')

# --- Gráfico de vendas filtradas ---
st.subheader('Vendas por Data (Filtrado)')
vendas_filtradas_por_data = df_filtrado.groupby('data')['total_venda'].sum()

fig4, ax4 = plt.subplots()
sns.lineplot(x=vendas_filtradas_por_data.index, y=vendas_filtradas_por_data.values, ax=ax4)
ax4.set_title('Vendas por Data (Filtrado)')
ax4.set_ylabel('Total em Vendas')
ax4.set_xlabel('Data')
st.pyplot(fig4)

# --- KPIs Dinâmicos ---
st.subheader("KPIs Filtrados")
colf1, colf2, colf3 = st.columns(3)
colf1.metric("Total Filtrado", f"R$ {df_filtrado['total_venda'].sum():,.2f}")
colf2.metric("Quantidade", df_filtrado['quantidade'].sum())
colf3.metric("Preço Médio", f"R$ {df_filtrado['preco_unitario'].mean():.2f}")

# --- Previsão de Vendas ---
st.header("📈 Previsão de Vendas (Regressão Linear)")
vendas_diarias = df.groupby('data')['total_venda'].sum().reset_index()
vendas_diarias['dias'] = (vendas_diarias['data'] - vendas_diarias['data'].min()).dt.days

X = vendas_diarias[['dias']]
y = vendas_diarias['total_venda']
model = LinearRegression().fit(X, y)

dias_futuro = np.array([[x] for x in range(vendas_diarias['dias'].max()+1, vendas_diarias['dias'].max()+31)])
previsao = model.predict(dias_futuro)

datas_futuras = pd.date_range(start=vendas_diarias['data'].max() + pd.Timedelta(days=1), periods=30)

fig5, ax5 = plt.subplots()
ax5.plot(vendas_diarias['data'], y, label='Vendas reais')
ax5.plot(datas_futuras, previsao, label='Previsão', linestyle='--')
ax5.set_title('Previsão de Vendas para os próximos 30 dias')
ax5.set_ylabel('Total em Vendas')
ax5.set_xlabel('Data')
ax5.legend()
st.pyplot(fig5)
