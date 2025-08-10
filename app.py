import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta
import re
from pathlib import Path

# ==================== CONFIG ====================
st.set_page_config(page_title="Análise de Preços Skyscanner", layout="wide", initial_sidebar_state="expanded")

# ---- LOGO no topo (procura no repo)
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "skyscanner.png" if (ASSETS / "skyscanner.png").exists() else ROOT / "skyscanner.png"
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=210)
else:
    st.title("Análise de Preços Skyscanner")

# ==================== ESTILO (Sidebar azul) ====================
st.markdown("""
<style>
:root{
  --sky-blue:#0B5FFF; --sky-blue-dark:#0A2A6B; --sky-blue-500:#1E6BFF;
  --sky-blue-300:#7FADFF; --sky-blue-200:#A5C3FF; --sky-blue-100:#E6F0FF;
  --text:#0f172a; --muted:#e5e7eb;
}
.stApp{background:#fff;color:var(--text);}
h1,h2,h3{color:var(--sky-blue-dark)!important;}

/* Barra de controles no topo */
.topbar {
  display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;
  padding:.75rem 0; border-top:1px solid var(--muted); border-bottom:1px solid var(--muted);
}
.topbar .label { font-weight:700; color:var(--sky-blue-dark); }

/* Sidebar */
section[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid var(--muted);}
section[data-testid="stSidebar"] *{color:var(--sky-blue-dark)!important;}

/* Inputs */
section[data-testid="stSidebar"] div[data-baseweb="select"]>div,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stDateInput input{
  border:1px solid var(--muted); box-shadow:none;
}

/* Chips MULTISELECT: força azul */
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"],
section[data-testid="stSidebar"] [data-baseweb="tag"],
section[data-testid="stSidebar"] [class*="tagRoot"]{
  background:var(--sky-blue)!important; color:#fff!important; border-color:var(--sky-blue-dark)!important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] *,
section[data-testid="stSidebar"] [class*="tagRoot"] *{
  color:#fff!important; fill:#fff!important; stroke:#fff!important;
}

/* Calendário azul */
div[data-baseweb="calendar"] [aria-selected="true"], div[data-baseweb="calendar"] .selected{
  background:var(--sky-blue)!important; color:#fff!important;
}
div[data-baseweb="calendar"] button:hover{ background:var(--sky-blue-200)!important; }

/* Slider ADVP azul */
div[data-testid="stSlider"] [data-baseweb="slider"]>div>div{ background:var(--sky-blue-100)!important; }
div[data-testid="stSlider"] [data-baseweb="slider"]>div>div:nth-child(2){ background:var(--sky-blue)!important; }
div[data-testid="stSlider"] [role="slider"]{
  background:#fff!important; border:3px solid var(--sky-blue)!important;
  box-shadow:0 0 0 2px rgba(11,95,255,.15)!important;
}

/* Toasts/alerts */
div[data-testid="stAlert"], div[data-baseweb="toast"], div[data-testid="stException"]{
  background:var(--sky-blue)!important; color:#fff!important; border:1px solid var(--sky-blue-dark)!important;
}
div[data-testid="stAlert"] *, div[data-baseweb="toast"] *, div[data-testid="stException"] *{color:#fff!important;}
</style>
""", unsafe_allow_html=True)

# ==================== PALETAS ====================
BLUES = ['#0A2A6B','#0B5FFF','#1E6BFF','#3880FF','#5A97FF','#7FADFF','#A5C3FF','#CAD9FF','#E6F0FF']
DARK_NAVY = '#0A2A6B'
PRIMARY_BLUE = '#0B5FFF'
COLOR_123 = '#003399'
COLOR_MAX = '#00B3FF'
COLOR_MELHOR = '#00F7FF'  # fluorescente para "Melhor Preço"
COLOR_INCREASING = '#1E6BFF'
COLOR_DECREASING = '#A5C3FF'

# Paletas temporais (AZUL ↔ CINZA)
SERIES_BLUES = ['#0B5FFF', '#3880FF', '#1E6BFF', '#7FADFF']
SERIES_GRAYS = ['#4B5563', '#9CA3AF', '#6B7280', '#94A3B8']

def build_blue_gray_map(categories):
    cats = list(categories)
    cmap = {}
    if 'Melhor Preço' in cats:  # mantém fluorescente
        cmap['Melhor Preço'] = COLOR_MELHOR
        cats.remove('Melhor Preço')
    b = g = 0; blue_turn = True
    for c in cats:
        if blue_turn:
            cmap[c] = SERIES_BLUES[b % len(SERIES_BLUES)]; b += 1
        else:
            cmap[c] = SERIES_GRAYS[g % len(SERIES_GRAYS)]; g += 1
        blue_turn = not blue_turn
    return cmap

largura_barras_precos = 0.8
chart_height = 400
chart_height_cascade = 560
ADVPS_ORDEM = [1,3,7,14,21,30,60,90]

def theme_plotly(fig):
    fig.update_layout(
        template='plotly_white', paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color=DARK_NAVY, size=14),
        legend=dict(bgcolor='rgba(255,255,255,0.85)'),
        hoverlabel=dict(bgcolor=PRIMARY_BLUE, font_color='white')
    )
    return fig

# ==================== FORMATADORES (datas e números BR) ====================
def fmt_int_br(n:int) -> str:
    return f"{int(n):,}".replace(",", "¤").replace(".", ",").replace("¤", ".")

def format_data_br_swap_if_ambiguous(ts) -> str:
    if pd.isna(ts): return ""
    ts = pd.to_datetime(ts, errors='coerce')
    if pd.isna(ts): return ""
    d, m = int(ts.day), int(ts.month)
    if d <= 12 and m <= 12:
        try:
            ts = ts.replace(day=m, month=d)
        except ValueError:
            pass
    return ts.strftime("%d/%m/%Y %H:%M")

def format_dates_in_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].apply(lambda x: format_data_br_swap_if_ambiguous(x) if pd.notna(x) else "")
            continue
        if pd.api.types.is_period_dtype(d[c]):
            d[c] = d[c].dt.to_timestamp()
            d[c] = d[c].apply(lambda x: format_data_br_swap_if_ambiguous(x) if pd.notna(x) else "")
            continue
        if pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
            if d[c].map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64, datetime, np.datetime64))).any():
                d[c] = pd.to_datetime(d[c], errors="coerce")
                d[c] = d[c].apply(lambda x: format_data_br_swap_if_ambiguous(x) if pd.notna(x) else "")
    return d

# ==================== REGIÕES ====================
def expand_bidirectional(pairs):
    s=set()
    for t in pairs:
        t=(t or "").strip()
        if not t: continue
        s.add(t)
        if '-' in t:
            a,b=t.split('-',1); s.add(f"{b}-{a}")
    return sorted(s)

REGIOES_RAW={
 "NORTE":["BEL-GRU","BEL-GIG","BEL-GRU","BEL-MCP","BEL-STM","BEL-FOR","BEL-MAO","BEL-REC","BEL-CWB","BEL-FLN","BEL-CNF","BEL-NVT","BEL-SDU","CKS-CNF","MAO-STM","MAO-TBT","MAO-VCP","MAO-REC","MAO-PVH","MAO-TFF","FOR-MAO"],
 "NORDESTE":["AJU-GRU","AJU-GIG","AJU-VCP","AJU-CGH","AJU-CNF","BPS-CNF","BPS-CGH","BPS-GRU","FOR-GRU","FOR-GIG","FOR-REC","FOR-VCP","FOR-SSA","GYN-MCZ","GYN-REC","GYN-VCP","GYN-SDU","JDO-VCP","MCZ-VCP","PNZ-VCP","REC-SSA","REC-VCP","REC-VIX","SSA-VCP","SSA-VIX"],
 "CENTRO-OESTE":["BSB-CGH","BSB-REC","BSB-SDU","BSB-GIG","BSB-SSA","BSB-VCP","BSB-CNF","BSB-GRU","BSB-NAT","BSB-THE","BSB-SLZ","BSB-FOR","BSB-CGB","BSB-CWB","BSB-VIX","BSB-JPA","CGB-GRU","CGR-GRU","CGR-VCP"],
 "SUDESTE":["CAC-GRU","CGH-SDU","CGH-SSA","CGH-REC","CGH-CNF","CGH-CWB","CGH-POA","CGH-FLN","CGH-GYN","CGH-NVT","CGH-FOR","CGH-MCZ","CGH-VIX","CGH-GIG","CGH-THE","CGH-JPA","CGH-NAT","CGH-CGR","CNF-SSA","CNF-GIG","CNF-GRU","CNF-REC","CNF-FOR","CNF-SLZ","CNF-MAO","CNF-CWB","CNF-FLN","CNF-VCP","CNF-SLZ","CNF-MCZ","CNF-NAT","CNF-VIX","CNF-POA","CNF-THE"],
 "SUL":["CWB-GIG","CWB-MAO","CWB-SSA","CWB-POA","CWB-IGU","CWB-REC","CWB-SDU","FLN-GIG","FLN-SDU","FLN-MAO","FLN-SSA"]
}
REGIOES_TRECHOS={k:expand_bidirectional(v) for k,v in REGIOES_RAW.items()}

# ==================== NORMALIZAÇÃO TRECHO ====================
def normalize_trecho(value:str)->str:
    if value is None: return ""
    s=str(value).upper().strip().replace('—','-').replace('–','-').replace('/','-')
    s=re.sub(r'\s+','',s)
    m=re.findall(r'[A-Z]{3}',s)
    if len(m)>=2: return f"{m[0]}-{m[1]}"
    s=re.sub(r'[^A-Z]','-',s); s=re.sub(r'-+','-',s).strip('-'); return s

def normalize_set(items): return {normalize_trecho(x) for x in items}
REGIOES_TRECHOS_STD={k:normalize_set(v) for k,v in REGIOES_TRECHOS.items()}

# ==================== HELPERS ====================
def build_color_map(categories, include_named=True):
    named={'Melhor Preço': COLOR_MELHOR,'Grupo123': PRIMARY_BLUE,'123MILHAS': COLOR_123,'MAXMILHAS': COLOR_MAX} if include_named else {}
    cmap={}; base=[c for c in BLUES if c not in named.values()]; i=0
    for c in categories:
        if c in named: cmap[c]=named[c]
        else: cmap[c]=base[i % len(base)]; i+=1
    return cmap

def _ensure_dataframe(obj): return obj.to_frame().T if isinstance(obj,pd.Series) else obj
def _show_styled_table(df_table, fmt='{:,.0f}', cmap='Blues'):
    df_table=_ensure_dataframe(df_table)
    if df_table is None or df_table.empty or df_table.shape[1]==0:
        st.dataframe(df_table); return
    d=format_dates_in_df_for_display(df_table.copy())
    d.columns=d.columns.map(str)
    st.dataframe(d.style.background_gradient(cmap=cmap).format(fmt))

# ========= Agrupadores de período (alinhados ao filtro) =========
def add_period_column(df: pd.DataFrame, modo: str, sd, ed) -> pd.DataFrame:
    """
    Alinha períodos exatamente ao intervalo do filtro:
      - Semanal: blocos de 7 dias
      - Quinzenal: blocos de 15 dias
      - Mensal: blocos de 30 dias
    Rótulo = fim do bloco (truncado em ed). Nada fora [sd, ed].
    """
    d = df.dropna(subset=['Data/Hora da Busca']).copy()
    dt = d['Data/Hora da Busca'].dt.floor('D')

    sd = pd.to_datetime(sd).normalize()
    ed = pd.to_datetime(ed).normalize()

    step = 7 if modo == 'Semanal' else (15 if modo == 'Quinzenal' else 30)

    # restringe ao intervalo primeiro
    mask = (dt >= sd) & (dt <= ed)
    d = d.loc[mask].copy()
    dt = d['Data/Hora da Busca'].dt.floor('D')

    # índice do bloco a partir de sd
    offset_days = (dt - sd).dt.days
    idx = (offset_days // step).clip(lower=0)

    bin_start = sd + pd.to_timedelta(idx * step, unit='D')
    bin_end   = bin_start + pd.to_timedelta(step - 1, unit='D')
    d['PERIODO'] = np.where(bin_end <= ed, bin_end, ed)
    d['PERIODO'] = pd.to_datetime(d['PERIODO']).dt.normalize()
    return d

def line_fig(df, x, y, color, title, percent=False, height=380, cmap=None):
    cmap = cmap or {}
    fig = px.line(df, x=x, y=y, color=color, markers=True, color_discrete_map=cmap, title=title)
    if percent: fig.update_yaxes(ticksuffix='%')
    fig.update_layout(height=height, legend_title=None, xaxis_title=None)
    return theme_plotly(fig)

def top3_competitors(df: pd.DataFrame, exclude=('123MILHAS','MAXMILHAS')) -> list:
    comp = df[~df['Agência/Companhia'].isin(exclude)]
    if comp.empty: return []
    return list(comp.groupby('Agência/Companhia').size().sort_values(ascending=False).head(3).index)

def apply_filters_for_timeseries(df_regiao, tipo_agencia_filtro, advp_valor, advp_range,
                                 datas_sel, trecho_sel, agencias_para_analise, cias):
    d = df_regiao.copy()
    # mantém 123 e MAX separados nas séries
    if tipo_agencia_filtro=='Agências':
        d = d[~d['Agência/Companhia'].isin(cias)]
    elif tipo_agencia_filtro=='Cias':
        d = d[d['Agência/Companhia'].isin(cias+['123MILHAS','MAXMILHAS'])]

    if advp_valor!='Todos':
        d = d[d['ADVP']==advp_valor]
    else:
        d = d[(d['ADVP']>=advp_range[0])&(d['ADVP']<=advp_range[1])]

    if len(datas_sel)==2:
        sd,ed=pd.to_datetime(datas_sel[0]),pd.to_datetime(datas_sel[1])
        d = d[(d['Data/Hora da Busca'].dt.date>=sd.date())&(d['Data/Hora da Busca'].dt.date<=ed.date())]

    if trecho_sel!='Todos os Trechos':
        d = d[d['TRECHO']==trecho_sel]

    alvo = set(agencias_para_analise) | {'123MILHAS','MAXMILHAS'}
    d = d[d['Agência/Companhia'].isin([x for x in d['Agência/Companhia'].unique() if x in alvo])]
    return d

# ==================== CARGA (A..M) ====================
PARQUET_REPO = Path(__file__).parent / "OFERTAS.parquet"
PARQUET_LOCAL = Path(r"C:\Users\tassiana.silva\OneDrive - 123 VIAGENS E TURISMO LTDA\SKYSCANNER02\skyscanner-app\OFERTAS.parquet")
caminho_arquivo = PARQUET_LOCAL if PARQUET_LOCAL.exists() else PARQUET_REPO

@st.cache_data
def carregar_dados(caminho):
    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}"); return None
    try:
        df=pd.read_parquet(caminho)
        expected=['Nome do Arquivo','Companhia Aérea','Horário1','Horário2','Horário3','Tipo de Voo','Data do Voo','Data/Hora da Busca','Agência/Companhia','Preço','TRECHO','ADVP','RANKING']
        if df.shape[1] < len(expected):
            st.error(f"O arquivo tem {df.shape[1]} colunas, esperado ≥ {len(expected)} (A..M)."); return None
        new_cols=list(df.columns)
        for i,n in enumerate(expected): new_cols[i]=n
        df.columns=new_cols

        for c in ['Data/Hora da Busca','Data do Voo','Horário1','Horário2','Horário3']:
            if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c]=pd.to_datetime(df[c], errors='coerce', dayfirst=True)
            elif c in df.columns:
                df[c]=pd.to_datetime(df[c], errors='coerce')

        for c in ['Preço','ADVP','RANKING']:
            if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
        if 'TRECHO' in df.columns: df['TRECHO_STD']=df['TRECHO'].map(normalize_trecho)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}"); return None

df=carregar_dados(caminho_arquivo)

# ==================== APP ====================
if df is not None and not df.empty:

    # -------- Sidebar
    st.sidebar.header("Filtros")
    st.sidebar.subheader("Filtro por Região")
    regiao_sel=st.sidebar.selectbox("Região", ['Todas']+list(REGIOES_TRECHOS_STD.keys()), index=0)

    df_regiao=df.copy()
    if regiao_sel!='Todas':
        df_regiao=df_regiao[df_regiao['TRECHO_STD'].isin(REGIOES_TRECHOS_STD[regiao_sel])]

    if regiao_sel=='Todas':
        trechos_disp=sorted(df_regiao['TRECHO'].dropna().unique())
    else:
        std_set=REGIOES_TRECHOS_STD[regiao_sel]
        trechos_disp=sorted(df_regiao.loc[df_regiao['TRECHO_STD'].isin(std_set),'TRECHO'].dropna().unique())
        if not trechos_disp: trechos_disp=sorted(list(REGIOES_TRECHOS[regiao_sel]))

    st.sidebar.subheader("Análise 123/Max")
    config_123_max_filtro=st.sidebar.selectbox("Como analisar 123MILHAS e MAXMILHAS?", ("Separado","Grupo123"))
    st.sidebar.markdown("---")

    tipo_agencia_filtro=st.sidebar.selectbox("Filtro de Agências/Cias", ("Geral","Agências","Cias"))
    cias=['GOL','LATAM','AZUL','JETSMART','TAP']
    agencias=[a for a in df_regiao['Agência/Companhia'].dropna().unique() if a not in cias]

    if tipo_agencia_filtro=='Agências':
        todas_agencias=sorted(agencias)
    elif tipo_agencia_filtro=='Cias':
        base=cias.copy()
        for a in ['123MILHAS','MAXMILHAS']:
            if a in df_regiao['Agência/Companhia'].dropna().unique(): base.append(a)
        todas_agencias=sorted(base)
    else:
        todas_agencias=sorted(df_regiao['Agência/Companhia'].dropna().unique())

    principais_default=[x for x in ['123MILHAS','MAXMILHAS'] if x in todas_agencias]
    agencias_principais=st.sidebar.multiselect("Agência(s) Principal(is)", todas_agencias, default=principais_default)
    concorrentes_pool=[a for a in todas_agencias if a not in agencias_principais]
    agencias_concorrentes=st.sidebar.multiselect("Agência(s) Concorrente(s)", ['Todos']+concorrentes_pool, default=['Todos'])
    agencias_para_analise=agencias_principais + (concorrentes_pool if 'Todos' in agencias_concorrentes else agencias_concorrentes)

    trecho_sel=st.sidebar.selectbox("Trecho", ['Todos os Trechos']+trechos_disp)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro por ADVP")
    if df_regiao.empty or df_regiao['ADVP'].dropna().empty:
        advp_min,advp_max=0,1
    else:
        advp_min,advp_max=int(df_regiao['ADVP'].min()), int(df_regiao['ADVP'].max())
    range_default=(advp_min,advp_max) if advp_min<advp_max else (advp_min,advp_min+1)
    advp_valor=st.sidebar.selectbox('Valor fixo de ADVP', options=['Todos']+ADVPS_ORDEM, index=0)
    advp_range=st.sidebar.slider('Ou intervalo de ADVP', min_value=advp_min, max_value=max(advp_max,advp_min+1), value=range_default)

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros de Data")
    df_dt=df_regiao.dropna(subset=['Data/Hora da Busca'])
    if not df_dt.empty:
        dmin,dmax=df_dt['Data/Hora da Busca'].min().date(), df_dt['Data/Hora da Busca'].max().date()
    else:
        dmin=dmax=datetime.now().date()
    periodo=st.sidebar.selectbox('Período', ('Últimos 7 dias','Últimos 15 dias','Últimos 30 dias','Período Personalizado'))
    if periodo=='Últimos 7 dias': start_default=max(dmax - timedelta(days=7), dmin)
    elif periodo=='Últimos 15 dias': start_default=max(dmax - timedelta(days=15), dmin)
    elif periodo=='Últimos 30 dias': start_default=max(dmax - timedelta(days=30), dmin)
    else: start_default=dmin
    datas_sel=st.sidebar.date_input('Intervalo de datas', value=(start_default, dmax), min_value=dmin, max_value=dmax)

    # -------- Aplicação de filtros principal (para seções 1..8)
    df_filtrado=df_regiao.copy()
    if tipo_agencia_filtro=='Agências':
        df_filtrado=df_filtrado[~df_filtrado['Agência/Companhia'].isin(cias)]
    elif tipo_agencia_filtro=='Cias':
        df_filtrado=df_filtrado[df_filtrado['Agência/Companhia'].isin(cias+['123MILHAS','MAXMILHAS'])]

    if config_123_max_filtro=='Grupo123':
        df_filtrado=df_filtrado.copy()
        df_filtrado['Agência/Companhia']=df_filtrado['Agência/Companhia'].replace(['123MILHAS','MAXMILHAS'],'Grupo123')

    if advp_valor!='Todos':
        df_filtrado=df_filtrado[df_filtrado['ADVP']==advp_valor]
    else:
        df_filtrado=df_filtrado[(df_filtrado['ADVP']>=advp_range[0])&(df_filtrado['ADVP']<=advp_range[1])]

    df_filtrado=df_filtrado[df_filtrado['Agência/Companhia'].isin(agencias_para_analise)]

    if len(datas_sel)==2:
        sd,ed=pd.to_datetime(datas_sel[0]),pd.to_datetime(datas_sel[1])
        df_filtrado=df_filtrado[(df_filtrado['Data/Hora da Busca'].dt.date>=sd.date())&(df_filtrado['Data/Hora da Busca'].dt.date<=ed.date())]

    if trecho_sel!='Todos os Trechos':
        df_filtrado=df_filtrado[df_filtrado['TRECHO']==trecho_sel]

    # ========================= 1. Preços por Agência =========================
    if not df_filtrado.empty:
        st.header("1. Comparação de Preços por Agência")
        c1,c2=st.columns(2)

        with c1:
            st.subheader("Agências Principais")
            alvo=(['Grupo123'] if config_123_max_filtro=='Grupo123' else agencias_principais)
            df_princ=df_filtrado[df_filtrado['Agência/Companhia'].isin(alvo)]
            if not df_princ.empty:
                pm=df_princ.groupby('Agência/Companhia',as_index=False)['Preço'].mean()
                cmap=build_color_map(pm['Agência/Companhia'].unique())
                fig=px.bar(pm,x='Agência/Companhia',y='Preço',text='Preço',title='Preço Médio',
                           labels={'Agência/Companhia':'Agência','Preço':'Preço Médio (R$)'},
                           color='Agência/Companhia', color_discrete_map=cmap)
                fig.update_layout(yaxis_title='Preço Médio (R$)', height=chart_height)
                fig.update_traces(texttemplate='<b>R$ %{y:,.0f}</b>', textposition='inside',
                                  insidetextfont=dict(size=18,color='white'), width=largura_barras_precos)
                theme_plotly(fig); st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados das principais.")

        with c2:
            st.subheader("Agências Concorrentes")
            if config_123_max_filtro=='Grupo123':
                df_conc=df_filtrado[df_filtrado['Agência/Companhia']!='Grupo123']
            else:
                df_conc=df_filtrado[~df_filtrado['Agência/Companhia'].isin(agencias_principais)]
            if not df_conc.empty:
                pm=df_conc.groupby('Agência/Companhia',as_index=False)['Preço'].mean().sort_values('Preço')
                cmap=build_color_map(pm['Agência/Companhia'].unique(), include_named=False)
                fig=px.bar(pm,x='Agência/Companhia',y='Preço',text='Preço',title='Preço Médio',
                           labels={'Agência/Companhia':'Agência','Preço':'Preço Médio (R$)'},
                           color='Agência/Companhia', color_discrete_map=cmap)
                fig.update_layout(yaxis_title='Preço Médio (R$)', height=chart_height)
                fig.update_traces(texttemplate='<b>R$ %{y:,.0f}</b>', textposition='inside',
                                  insidetextfont=dict(size=18,color='white'), width=largura_barras_precos)
                theme_plotly(fig); st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados de concorrentes.")

        # ========================= 2. Gauges =========================
        st.header("2. Comparativo de Preços vs. Melhor Concorrente")
        if config_123_max_filtro=='Grupo123':
            df_comp=df_filtrado[df_filtrado['Agência/Companhia']!='Grupo123']
            pm_grupo=df_filtrado.loc[df_filtrado['Agência/Companhia']=='Grupo123','Preço'].mean()
            if not df_comp.empty:
                melhor=df_comp.groupby('Agência/Companhia')['Preço'].mean().min()
                if pd.notna(melhor):
                    diff=((pm_grupo-melhor)/melhor)*100 if pd.notna(pm_grupo) else 0
                    fig=go.Figure(go.Indicator(mode="gauge+number", value=diff,
                        title={'text':"Grupo123 vs. Melhor Concorrente (%)"},
                        number={'suffix':'%','valueformat':'.0f'},
                        gauge={'axis':{'range':[-50,50]},'bar':{'color':PRIMARY_BLUE},'bgcolor':'white'}))
                    fig.update_layout(height=chart_height); theme_plotly(fig); st.plotly_chart(fig, use_container_width=True)
        else:
            df_comp=df_filtrado[~df_filtrado['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS'])]
            if not df_comp.empty:
                melhor=df_comp.groupby('Agência/Companhia')['Preço'].mean().min()
                pm_123=df_filtrado.loc[df_filtrado['Agência/Companhia']=='123MILHAS','Preço'].mean()
                pm_max=df_filtrado.loc[df_filtrado['Agência/Companhia']=='MAXMILHAS','Preço'].mean()
                if pd.notna(melhor):
                    d123=((pm_123-melhor)/melhor)*100 if pd.notna(pm_123) else 0
                    dmax=((pm_max-melhor)/melhor)*100 if pd.notna(pm_max) else 0
                    g1,g2=st.columns(2)
                    with g1:
                        fig=go.Figure(go.Indicator(mode="gauge+number", value=d123,
                            title={'text':"123MILHAS vs. Melhor Concorrente (%)"},
                            number={'suffix':'%','valueformat':'.0f'},
                            gauge={'axis':{'range':[-50,50]},'bar':{'color':COLOR_123},'bgcolor':'white'}))
                        fig.update_layout(height=chart_height); theme_plotly(fig); st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        fig=go.Figure(go.Indicator(mode="gauge+number", value=dmax,
                            title={'text':"MAXMILHAS vs. Melhor Concorrente (%)"},
                            number={'suffix':'%','valueformat':'.0f'},
                            gauge={'axis':{'range':[-50,50]},'bar':{'color':COLOR_MAX},'bgcolor':'white'}))
                        fig.update_layout(height=chart_height); theme_plotly(fig); st.plotly_chart(fig, use_container_width=True)

        # ========================= 3. Participação nos Rankings =========================
        st.header("3. Análise de Participação nos Rankings")
        counts=df_filtrado.groupby(['Agência/Companhia','RANKING']).size().unstack(fill_value=0)
        if 1 in counts.columns or '1' in counts.columns:
            chave=1 if 1 in counts.columns else '1'
            counts=counts.sort_values(by=chave, ascending=False)
        counts_tot=counts.copy(); counts_tot['Total']=counts_tot.sum(axis=1); counts_tot.loc['Total']=counts_tot.sum(numeric_only=True,axis=0)

        st.subheader("Quantidade de Ofertas por Ranking (com Totais)")
        core_rows,core_cols=counts.index,counts.columns
        col_totais=counts.sum(axis=0).replace(0,pd.NA)
        gmap=counts[core_cols].div(col_totais,axis=1).fillna(0).reindex(index=core_rows,columns=core_cols)
        st.dataframe(format_dates_in_df_for_display(counts_tot).style.format('{:,.0f}').background_gradient(
            cmap='Blues', gmap=gmap, subset=pd.IndexSlice[core_rows,core_cols], axis=None))

        row_sums=counts.sum(axis=1).replace(0,float('nan'))
        pct_row=(counts.divide(row_sums,axis=0)*100).fillna(0).round(2)
        col_r1=1 if 1 in pct_row.columns else ('1' if '1' in pct_row.columns else None)
        if col_r1 is not None: pct_row=pct_row.sort_values(by=col_r1, ascending=False)
        st.subheader("Participação (%) por Ranking – dentro da Agência (linha)")
        _show_styled_table(pct_row, fmt='{:.2f}%', cmap='Blues')

        col_sums=counts.sum(axis=0).replace(0,float('nan'))
        pct_col=(counts.divide(col_sums,axis=1)*100).fillna(0).round(2)
        st.subheader("Participação (%) por Ranking – dentro do Ranking (coluna)")
        _show_styled_table(pct_col, fmt='{:.2f}%', cmap='Blues')

        # ========================= 4. Linhas por período do dia =========================
        st.header("4. Ranking de Melhor Preço por Período do Dia")
        hora_col='Horário1' if 'Horário1' in df_filtrado.columns else None
        if hora_col:
            df_h=df_filtrado.copy(); df_h[hora_col]=pd.to_datetime(df_h[hora_col], errors='coerce'); df_h=df_h.dropna(subset=[hora_col])
            if not df_h.empty:
                df_h['Hora do Voo']=df_h[hora_col].dt.hour
                if config_123_max_filtro=='Grupo123':
                    agencias_linhas=['Grupo123'] if 'Grupo123' in df_h['Agência/Companhia'].unique() else []
                else:
                    agencias_linhas=[a for a in agencias_principais if a in df_h['Agência/Companhia'].unique()]
                idx_min=df_h.groupby('Hora do Voo')['Preço'].idxmin()
                melhor=df_h.loc[idx_min,['Hora do Voo','Agência/Companhia','Preço']].copy()
                melhor.rename(columns={'Agência/Companhia':'Agência Melhor','Preço':'Preço (R$)'}, inplace=True)
                melhor['Agência']='Melhor Preço'
                linhas=[]
                for ag in agencias_linhas:
                    d=(df_h[df_h['Agência/Companhia']==ag].groupby('Hora do Voo')['Preço'].min().reset_index().rename(columns={'Preço':'Preço (R$)'}))
                    d['Agência']=ag; d['Agência Melhor']=ag; linhas.append(d)
                por_ag=pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame(columns=['Hora do Voo','Preço (R$)','Agência','Agência Melhor'])
                df_plot=pd.concat([melhor,por_ag], ignore_index=True).sort_values('Hora do Voo')
                df_plot['HoverAg']=np.where(df_plot['Agência']=='Melhor Preço', df_plot['Agência Melhor'], df_plot['Agência'])
                cmap=build_color_map(df_plot['Agência'].unique())
                periodos={'Madrugada':range(0,6),'Manhã':range(6,12),'Tarde':range(12,18),'Noite':range(18,24)}
                cols=st.columns(2); i=0
                for nome,horas in periodos.items():
                    d=df_plot[df_plot['Hora do Voo'].isin(horas)]
                    if d.empty: continue
                    fig=px.line(d,x='Hora do Voo',y='Preço (R$)',color='Agência',markers=True,
                                title=nome,color_discrete_map=cmap,custom_data=['HoverAg'])
                    fig.update_traces(hovertemplate="%{fullData.name}<br>Hora: %{x}<br>Preço: R$ %{y:.2f}<br>Agência: %{customdata[0]}")
                    fig.update_layout(height=chart_height,xaxis=dict(dtick=1),yaxis_title='Preço (R$)')
                    theme_plotly(fig)
                    with cols[i%2]: st.plotly_chart(fig, use_container_width=True)
                    i+=1
            else:
                st.info("Sem horários válidos após aplicar os filtros.")
        else:
            st.warning("Não há coluna de horário para esta análise.")

        # ========================= 5. Vantagem Competitiva por Trecho =========================
        st.header("5. Análise de Vantagem Competitiva por Trecho")
        base=df_filtrado[['TRECHO','Data/Hora da Busca','Agência/Companhia','Preço','RANKING']]
        base=base[base['RANKING'].isin([1,2,3])].copy()
        if not base.empty:
            pv=base.pivot_table(index=['TRECHO','Data/Hora da Busca'], columns='RANKING',
                                values=['Preço','Agência/Companhia'], aggfunc='first').reset_index()
            pv.columns=['_'.join(map(str,c)).strip() for c in pv.columns.values]
            pv.rename(columns={'TRECHO_':'TRECHO','Data/Hora da Busca_':'Data/Hora da Busca',
                               'Preço_1':'Preço_1','Preço_2':'Preço_2','Preço_3':'Preço_3',
                               'Agência/Companhia_1':'Agência_1','Agência/Companhia_2':'Agência_2','Agência/Companhia_3':'Agência_3'}, inplace=True)
            pv.dropna(subset=['Preço_2','Agência_2'], inplace=True)
            pv=pv[pv['Agência_1']!=pv['Agência_2']]
            if not pv.empty:
                pv['Diferença_2_pct']=((pv['Preço_2']-pv['Preço_1'])/pv['Preço_1']*100).round(2)
                pv['Diferença_3_pct']=((pv['Preço_3']-pv['Preço_1'])/pv['Preço_1']*100).round(2)

                def tabela_top(ag):
                    from collections import Counter as C
                    d=pv[pv['Agência_1']==ag]
                    if d.empty: st.info(f"Sem dados para {ag}."); return
                    top=d.groupby('TRECHO').agg(
                        Vitorias=('TRECHO','count'),
                        Menor_Preco_1=('Preço_1','min'),
                        Menor_Preco_2=('Preço_2','min'),
                        Diferenca_Media_2=('Diferença_2_pct','mean'),
                        Menor_Preco_3=('Preço_3','min'),
                        Diferenca_Media_3=('Diferença_3_pct','mean')
                    ).sort_values('Diferenca_Media_2', ascending=False).head(20).reset_index()
                    seg=d.groupby('TRECHO')['Agência_2'].agg(lambda x:C(x).most_common(1)[0][0]).reset_index()
                    ter=d.groupby('TRECHO')['Agência_3'].agg(lambda x:C(x).most_common(1)[0][0]).reset_index()
                    top=top.merge(seg,on='TRECHO',how='left').merge(ter,on='TRECHO',how='left')
                    top=top[['TRECHO','Vitorias','Menor_Preco_1','Agência_2','Menor_Preco_2','Diferenca_Media_2','Agência_3','Menor_Preco_3','Diferenca_Media_3']].rename(
                        columns={'Menor_Preco_1':'Menor Preço 1º Lugar','Agência_2':'2º Lugar','Menor_Preco_2':'Menor Preço 2º Lugar',
                                 'Diferenca_Media_2':'Diferença (%) para 2º','Agência_3':'3º Lugar','Menor_Preco_3':'Menor Preço 3º Lugar',
                                 'Diferenca_Media_3':'Diferença (%) para 3º'})
                    st.subheader(f"Top 20 Trechos - {ag} (Vitórias)")
                    st.dataframe(format_dates_in_df_for_display(top).style.background_gradient(cmap='Blues').format({
                        'Menor Preço 1º Lugar':'R$ {:,.2f}','Menor Preço 2º Lugar':'R$ {:,.2f}','Menor Preço 3º Lugar':'R$ {:,.2f}',
                        'Diferença (%) para 2º':'{:.2f}%','Diferença (%) para 3º':'{:.2f}%'
                    }))
                if config_123_max_filtro=='Grupo123': tabela_top('Grupo123')
                else: tabela_top('123MILHAS'); tabela_top('MAXMILHAS')

        # ========================= 6. Cascata por ADVP =========================
        st.header("6. 123MILHAS & MAXMILHAS por ADVP (Cascata)")
        df_advp=df_regiao.copy()
        if tipo_agencia_filtro=='Agências': df_advp=df_advp[~df_advp['Agência/Companhia'].isin(cias)]
        elif tipo_agencia_filtro=='Cias':   df_advp=df_advp[df_advp['Agência/Companhia'].isin(cias+['123MILHAS','MAXMILHAS'])]
        df_advp=df_advp[df_advp['Agência/Companhia'].isin(agencias_para_analise+['123MILHAS','MAXMILHAS','Grupo123'])]
        if len(datas_sel)==2:
            sd,ed=pd.to_datetime(datas_sel[0]),pd.to_datetime(datas_sel[1])
            df_advp=df_advp[(df_advp['Data/Hora da Busca'].dt.date>=sd.date())&(df_advp['Data/Hora da Busca'].dt.date<=ed.date())]
        if trecho_sel!='Todos os Trechos': df_advp=df_advp[df_advp['TRECHO']==trecho_sel]
        if advp_valor!='Todos': df_advp=df_advp[df_advp['ADVP']==advp_valor]
        else: df_advp=df_advp[(df_advp['ADVP']>=advp_range[0])&(df_advp['ADVP']<=advp_range[1])]

        def diffs_por_advp(df_base, agencia_ref):
            rows=[]
            if df_base.empty: return pd.DataFrame(rows)
            advps_pres=sorted({int(x) for x in df_base['ADVP'].dropna().unique()})
            advps=[a for a in ADVPS_ORDEM if a in advps_pres]
            for advp in advps:
                d=df_base[df_base['ADVP']==advp]
                preco_ag=d.loc[d['Agência/Companhia']==agencia_ref,'Preço'].mean()
                d_comp=d[~d['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS','Grupo123'])]
                if pd.isna(preco_ag) or d_comp.empty: continue
                melhor=d_comp.groupby('Agência/Companhia')["Preço"].mean().min()
                if pd.isna(melhor) or melhor==0: continue
                diff=((preco_ag-melhor)/melhor)*100
                rows.append({'ADVP':str(advp),'DifPct':diff,'LabelPct':f"{diff:.2f}%"})
            return pd.DataFrame(rows)

        def waterfall_advp(df_base, agencia_ref, title):
            data=diffs_por_advp(df_base, agencia_ref)
            if data.empty: return None
            fig=go.Figure(go.Waterfall(
                x=list(data['ADVP']), measure=['relative']*len(data), y=list(data['DifPct']),
                text=list(data['LabelPct']), textposition='outside',
                connector={'line':{'color':'rgba(0,0,0,0.2)'}},
                increasing={'marker':{'color':COLOR_INCREASING}},
                decreasing={'marker':{'color':COLOR_DECREASING}},
            ))
            fig.update_traces(texttemplate='<b>%{text}</b>', textfont_size=30, cliponaxis=False)
            cat_order=[str(x) for x in ADVPS_ORDEM if str(x) in list(data['ADVP'])]
            fig.update_layout(
                title=title, height=chart_height_cascade,
                xaxis=dict(type='category', categoryorder='array', categoryarray=cat_order, tickfont=dict(size=18)),
                yaxis=dict(title='Diferença vs Melhor Concorrente (%)', tickformat=".2f", ticksuffix='%', tickfont=dict(size=18)),
                margin=dict(t=70,b=50)
            ); return theme_plotly(fig)

        f1=waterfall_advp(df_advp, '123MILHAS' if config_123_max_filtro!='Grupo123' else 'Grupo123',
                          "123MILHAS por ADVP (Cascata)" if config_123_max_filtro!='Grupo123' else "Grupo123 por ADVP (Cascata)")
        if f1 is not None: st.plotly_chart(f1, use_container_width=True)
        else: st.info("Sem dados para 123MILHAS/Grupo123 nos filtros atuais.")
        if config_123_max_filtro!='Grupo123':
            f2=waterfall_advp(df_advp,'MAXMILHAS',"MAXMILHAS por ADVP (Cascata)")
            if f2 is not None: st.plotly_chart(f2,use_container_width=True)
            else: st.info("Sem dados para MAXMILHAS nos filtros atuais.")

        # ========================= 7. Cascata por Região =========================
        st.header("7. Diferença vs Melhor Concorrente por Região (Cascata)")
        df_regioes=df.copy()
        if tipo_agencia_filtro=='Agências': df_regioes=df_regioes[~df_regioes['Agência/Companhia'].isin(cias)]
        elif tipo_agencia_filtro=='Cias':   df_regioes=df_regioes[df_regioes['Agência/Companhia'].isin(cias+['123MILHAS','MAXMILHAS'])]
        if config_123_max_filtro=='Grupo123':
            df_regioes=df_regioes.copy()
            df_regioes['Agência/Companhia']=df_regioes['Agência/Companhia'].replace(['123MILHAS','MAXMILHAS'],'Grupo123')
        if advp_valor!='Todos': df_regioes=df_regioes[df_regioes['ADVP']==advp_valor]
        else: df_regioes=df_regioes[(df_regioes['ADVP']>=advp_range[0])&(df_regioes['ADVP']<=advp_range[1])]
        if len(datas_sel)==2:
            sd,ed=pd.to_datetime(datas_sel[0]),pd.to_datetime(datas_sel[1])
            df_regioes=df_regioes[(df_regioes['Data/Hora da Busca'].dt.date>=sd.date())&(df_regioes['Data/Hora da Busca'].dt.date<=ed.date())]
        if trecho_sel!='Todos os Trechos': df_regioes=df_regioes[df_regioes['TRECHO']==trecho_sel]
        df_regioes=df_regioes[df_regioes['Agência/Companhia'].isin(agencias_para_analise+['123MILHAS','MAXMILHAS','Grupo123'])]

        def diffs_por_regiao(df_base, agencia_ref):
            rows=[]
            for reg,std_set in REGIOES_TRECHOS_STD.items():
                dreg=df_base[df_base['TRECHO_STD'].isin(std_set)]
                if dreg.empty: continue
                preco_ag=dreg.loc[dreg['Agência/Companhia']==agencia_ref,'Preço'].mean()
                d_comp=dreg[~dreg['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS','Grupo123'])]
                if pd.isna(preco_ag) or d_comp.empty: continue
                melhor=d_comp.groupby('Agência/Companhia')["Preço"].mean().min()
                if pd.isna(melhor) or melhor==0: continue
                diff=((preco_ag-melhor)/melhor)*100
                rows.append({'REGIÃO':reg,'DifPct':diff,'LabelPct':f"{diff:.2f}%"})
            return pd.DataFrame(rows)

        def waterfall_regioes(df_base, agencia_ref, title):
            data=diffs_por_regiao(df_base,agencia_ref)
            if data.empty: return None
            fig=go.Figure(go.Waterfall(
                x=list(data['REGIÃO']), measure=['relative']*len(data), y=list(data['DifPct']),
                text=list(data['LabelPct']), textposition='outside',
                connector={'line':{'color':'rgba(0,0,0,0.2)'}},
                increasing={'marker':{'color':COLOR_INCREASING}},
                decreasing={'marker':{'color':COLOR_DECREASING}},
            ))
            fig.update_traces(texttemplate='<b>%{text}</b>', textfont_size=28, cliponaxis=False)
            fig.update_layout(title=title, height=chart_height_cascade,
                              xaxis=dict(type='category',tickfont=dict(size=16)),
                              yaxis=dict(title='Diferença vs Melhor Concorrente (%)', tickformat=".2f", ticksuffix='%', tickfont=dict(size=18)),
                              margin=dict(t=70,b=60))
            return theme_plotly(fig)

        alvo_ag='Grupo123' if config_123_max_filtro=='Grupo123' else '123MILHAS'
        fR1=waterfall_regioes(df_regioes,alvo_ag,"Grupo123 por Região (Cascata)" if config_123_max_filtro=='Grupo123' else "123MILHAS por Região (Cascata)")
        if fR1 is not None: st.plotly_chart(fR1,use_container_width=True)
        else: st.info("Sem dados por região para 123MILHAS/Grupo123 nos filtros atuais.")
        if config_123_max_filtro!='Grupo123':
            fR2=waterfall_regioes(df_regioes,'MAXMILHAS',"MAXMILHAS por Região (Cascata)")
            if fR2 is not None: st.plotly_chart(fR2,use_container_width=True)
            else: st.info("Sem dados por região para MAXMILHAS nos filtros atuais.")

        # ========================= 8. Buscas vs Ofertas (vertical, % e rótulo interno bold) =========================
        st.header("8. Buscas vs Ofertas por Agência/Cia (filtros aplicados)")
        base = df_filtrado.copy()
        if base.empty:
            st.info("Sem dados filtrados.")
        else:
            ofertas = base.groupby('Agência/Companhia').size().sort_values(ascending=False)
            total_ofertas = int(ofertas.sum()) if len(ofertas)>0 else 0
            resumo = ofertas.to_frame('Ofertas')
            resumo['Participação (%)'] = (resumo['Ofertas'] / total_ofertas * 100).round(2) if total_ofertas>0 else 0.0

            ctab, cbar = st.columns([2,3])
            with ctab:
                st.dataframe(
                    format_dates_in_df_for_display(resumo).style.background_gradient(cmap='Blues', subset=['Ofertas'])
                    .format({'Participação (%)':'{:.2f}%'})
                )

            with cbar:
                if config_123_max_filtro=='Grupo123':
                    principais_plot = ['Grupo123'] if 'Grupo123' in resumo.index else []
                else:
                    principais_plot = [a for a in agencias_principais if a in resumo.index]

                left_df  = resumo.loc[resumo.index.intersection(principais_plot)][['Participação (%)']].copy()
                right_df = resumo[['Participação (%)']].head(3).copy()

                cmap_bars = build_color_map(list(set(list(left_df.index)+list(right_df.index))))

                gL, gR = st.columns(2)
                if not left_df.empty:
                    dfL = left_df.reset_index()
                    dfL['lbl'] = dfL['Participação (%)'].map(lambda v: f"{v:.2f}%")
                    figL = px.bar(
                        dfL, x='Agência/Companhia', y='Participação (%)',
                        text='lbl', color='Agência/Companhia',
                        color_discrete_map=cmap_bars, title='Principais'
                    )
                    figL.update_traces(
                        texttemplate='<b>%{text}</b>', textposition='inside',
                        textfont=dict(size=18, color='white')
                    )
                    figL.update_layout(
                        height=380, yaxis_title='% no Total', showlegend=False,
                        uniformtext_minsize=14, uniformtext_mode='show', title_font=dict(size=18)
                    )
                    theme_plotly(figL); gL.plotly_chart(figL, use_container_width=True)
                else:
                    gL.info("Sem principais nos filtros.")

                if not right_df.empty:
                    dfR = right_df.reset_index()
                    dfR['lbl'] = dfR['Participação (%)'].map(lambda v: f"{v:.2f}%")
                    figR = px.bar(
                        dfR, x='Agência/Companhia', y='Participação (%)',
                        text='lbl', color='Agência/Companhia',
                        color_discrete_map=cmap_bars, title='TOP-3 Geral'
                    )
                    figR.update_traces(
                        texttemplate='<b>%{text}</b>', textposition='inside',
                        textfont=dict(size=18, color='white')
                    )
                    figR.update_layout(
                        height=380, yaxis_title='% no Total', showlegend=False,
                        uniformtext_minsize=14, uniformtext_mode='show', title_font=dict(size=18)
                    )
                    theme_plotly(figR); gR.plotly_chart(figR, use_container_width=True)
                else:
                    gR.info("Sem TOP-3.")

        # ---- Rodapé
        st.markdown("---")
        ultima_raw = df['Data/Hora da Busca'].max() if 'Data/Hora da Busca' in df.columns else None
        if 'Nome do Arquivo' in df.columns and df['Nome do Arquivo'].notna().any():
            qtd_buscas = int(df['Nome do Arquivo'].nunique())
        elif 'Data/Hora da Busca' in df.columns:
            qtd_buscas = int(df['Data/Hora da Busca'].dt.floor('min').nunique())
        else:
            qtd_buscas = 0
        qtd_ofertas = int(len(df))
        if pd.notna(ultima_raw):
            st.caption(
                f"Última atualização do banco: **{format_data_br_swap_if_ambiguous(ultima_raw)}** • "
                f"Buscas: **{fmt_int_br(qtd_buscas)}** • Ofertas: **{fmt_int_br(qtd_ofertas)}**"
            )
        else:
            st.caption(f"Buscas: **{fmt_int_br(qtd_buscas)}** • Ofertas: **{fmt_int_br(qtd_ofertas)}**")

        # ========================= 9. VISÃO TEMPORAL(EM DESENVOLVIMENTO) (Semanal | Quinzenal | Mensal) =========================
        st.markdown('<div class="topbar"><span class="label">Visão temporal (EM CONSTRUÇÃO):</span></div>', unsafe_allow_html=True)
        visao = st.radio(" ", options=['Semanal','Quinzenal','Mensal'], index=0, horizontal=True, label_visibility="collapsed")
        st.markdown("")  # espaçamento

        # Base temporal alinhada ao filtro
        sd_ts = pd.to_datetime(datas_sel[0])
        ed_ts = pd.to_datetime(datas_sel[1])
        df_ts_base = apply_filters_for_timeseries(df_regiao, tipo_agencia_filtro, advp_valor, advp_range,
                                                  datas_sel, trecho_sel, agencias_para_analise, cias)
        if df_ts_base.empty:
            st.info("Sem dados para séries temporais com os filtros atuais.")
        else:
            df_ts = add_period_column(df_ts_base, visao, sd_ts, ed_ts)
            top3 = top3_competitors(df_ts)
            alvo_agencias = [a for a in ['123MILHAS','MAXMILHAS'] if a in df_ts['Agência/Companhia'].unique()] + top3

            # ---------- 9.1 Agências Principais VS Concorrentes — Preço Médio
            st.subheader("9.1 Agências Principais VS Concorrentes — Preço Médio por Período")
            g = (df_ts[df_ts['Agência/Companhia'].isin(alvo_agencias)]
                 .groupby(['PERIODO','Agência/Companhia'])['Preço'].mean().reset_index())
            cmap = build_blue_gray_map(g['Agência/Companhia'].unique())
            fig = line_fig(g, 'PERIODO', 'Preço', 'Agência/Companhia',
                           f"Média de Preço ({visao}) – 123, MAX e TOP-3", percent=False, cmap=cmap)
            fig.update_yaxes(title_text="Preço (R$)")
            st.plotly_chart(fig, use_container_width=True)

            # ---------- 9.2 Quantidade de Ofertas por Ranking (com Totais) – linhas
            st.subheader("9.2 Quantidade de Ofertas por Ranking (com Totais) – 123, MAX e TOP-3")
            gtot = (df_ts[df_ts['Agência/Companhia'].isin(alvo_agencias)]
                    .groupby(['PERIODO','Agência/Companhia']).size().reset_index(name='Ofertas'))
            cmap = build_blue_gray_map(gtot['Agência/Companhia'].unique())
            figt = line_fig(gtot, 'PERIODO', 'Ofertas', 'Agência/Companhia',
                            f"Total de Ofertas ({visao})", percent=False, cmap=cmap)
            figt.update_yaxes(title_text="Qtd Ofertas")
            st.plotly_chart(figt, use_container_width=True)

            tabs = st.tabs(["Ranking 1", "Ranking 2", "Ranking 3"])
            for rnk, t in zip([1,2,3], tabs):
                with t:
                    gr = (df_ts[(df_ts['RANKING']==rnk) & (df_ts['Agência/Companhia'].isin(alvo_agencias))]
                          .groupby(['PERIODO','Agência/Companhia']).size().reset_index(name='Ofertas'))
                    if gr.empty: st.info("Sem dados"); continue
                    cmap = build_blue_gray_map(gr['Agência/Companhia'].unique())
                    fgr = line_fig(gr, 'PERIODO', 'Ofertas', 'Agência/Companhia',
                                   f"Ofertas de Ranking {rnk} ({visao})", percent=False, cmap=cmap)
                    fgr.update_yaxes(title_text="Qtd Ofertas")
                    st.plotly_chart(fgr, use_container_width=True)

            # ---------- 9.3 Participação (%) por Ranking – dentro do Ranking (coluna)
            st.subheader("9.3 Participação (%) por Ranking – dentro do Ranking (coluna)")
            tabs2 = st.tabs(["Ranking 1", "Ranking 2", "Ranking 3"])
            for rnk, t in zip([1,2,3], tabs2):
                with t:
                    base_r = df_ts[df_ts['RANKING']==rnk]
                    if base_r.empty: st.info("Sem dados"); continue
                    cnt = base_r.groupby(['PERIODO','Agência/Companhia']).size().reset_index(name='Q')
                    tot = cnt.groupby('PERIODO')['Q'].sum().reset_index(name='TOT')
                    share = cnt.merge(tot, on='PERIODO')
                    share['Participação (%)'] = (share['Q']/share['TOT']*100).round(2)
                    share = share[share['Agência/Companhia'].isin(alvo_agencias)]
                    if share.empty: st.info("Sem dados"); continue
                    cmap = build_blue_gray_map(share['Agência/Companhia'].unique())
                    fsh = line_fig(share, 'PERIODO', 'Participação (%)', 'Agência/Companhia',
                                   f"Share no Ranking {rnk} ({visao})", percent=True, cmap=cmap)
                    st.plotly_chart(fsh, use_container_width=True)

            # ---------- 9.4 Ranking de Melhor Preço por Período do Dia (hora e data)
            st.subheader("9.4 Ranking de Melhor Preço por Período do Dia (hora e data)")
            dH = df_ts.copy()
            dH['HORA'] = dH['Data/Hora da Busca'].dt.floor('H')
            ag_series = []
            for ag in ['123MILHAS','MAXMILHAS']:
                if ag in dH['Agência/Companhia'].unique():
                    s = dH[dH['Agência/Companhia']==ag].groupby('HORA')['Preço'].min().reset_index()
                    s['Série'] = ag; ag_series.append(s)
            smin = dH.groupby('HORA')['Preço'].min().reset_index(); smin['Série'] = 'Melhor Preço'; ag_series.append(smin)
            if ag_series:
                dfHplot = pd.concat(ag_series, ignore_index=True)
                cmapH = build_blue_gray_map(dfHplot['Série'].unique())
                figH = line_fig(dfHplot, 'HORA', 'Preço', 'Série',
                                "Preço mínimo por hora (123, MAX e Melhor Preço)", percent=False, cmap=cmapH)
                figH.update_yaxes(title_text="Preço (R$)")
                st.plotly_chart(figH, use_container_width=True)
            else:
                st.info("Sem dados horários.")

            # ---------- 9.5 123MILHAS & MAXMILHAS por ADVP – linhas (diferença vs melhor)
            st.subheader("9.5 Diferença vs Melhor Concorrente por ADVP – 123 e MAX")
            def diff_vs_best_by_period_advp(df_in, ag):
                rows=[]
                if df_in.empty or ag not in df_in['Agência/Companhia'].unique(): return pd.DataFrame()
                advps = sorted([int(x) for x in df_in['ADVP'].dropna().unique()])
                for advp in advps:
                    d = df_in[df_in['ADVP']==advp]
                    if d.empty: continue
                    d = add_period_column(d, visao, sd_ts, ed_ts)
                    grp = d.groupby('PERIODO')
                    for per, dfp in grp:
                        pm_ag = dfp.loc[dfp['Agência/Companhia']==ag,'Preço'].mean()
                        comp = dfp[~dfp['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS'])]
                        if pd.isna(pm_ag) or comp.empty: continue
                        best = comp.groupby('Agência/Companhia')['Preço'].mean().min()
                        if pd.isna(best) or best==0: continue
                        rows.append({'PERIODO':per,'ADVP':str(advp),'Diferença (%)':(pm_ag-best)/best*100})
                return pd.DataFrame(rows)

            for ag in [a for a in ['123MILHAS','MAXMILHAS'] if a in df_ts['Agência/Companhia'].unique()]:
                dd = diff_vs_best_by_period_advp(df_ts, ag)
                if dd.empty: st.info(f"Sem dados para {ag}."); continue
                figd = line_fig(dd, 'PERIODO', 'Diferença (%)', 'ADVP',
                                f"{ag} – Diferença vs Melhor Concorrente por ADVP ({visao})",
                                percent=True, cmap=build_blue_gray_map(dd['ADVP'].unique()))
                st.plotly_chart(figd, use_container_width=True)

            # ---------- 9.6 Diferença vs Melhor Concorrente por Região – linhas
            st.subheader("9.6 Diferença vs Melhor Concorrente por Região – 123 e MAX")
            def diff_vs_best_by_period_region(df_in, ag):
                rows=[]
                if df_in.empty or ag not in df_in['Agência/Companhia'].unique(): return pd.DataFrame()
                d = add_period_column(df_in, visao, sd_ts, ed_ts)
                for reg,std_set in REGIOES_TRECHOS_STD.items():
                    dr = d[d['TRECHO_STD'].isin(std_set)]
                    if dr.empty: continue
                    grp = dr.groupby('PERIODO')
                    for per, dfp in grp:
                        pm_ag = dfp.loc[dfp['Agência/Companhia']==ag,'Preço'].mean()
                        comp = dfp[~dfp['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS'])]
                        if pd.isna(pm_ag) or comp.empty: continue
                        best = comp.groupby('Agência/Companhia')['Preço'].mean().min()
                        if pd.isna(best) or best==0: continue
                        rows.append({'PERIODO':per,'REGIÃO':reg,'Diferença (%)':(pm_ag-best)/best*100})
                return pd.DataFrame(rows)

            for ag in [a for a in ['123MILHAS','MAXMILHAS'] if a in df_ts['Agência/Companhia'].unique()]:
                dr = diff_vs_best_by_period_region(df_ts, ag)
                if dr.empty: st.info(f"Sem dados regionais para {ag}."); continue
                # pega até 5 regiões com maior volume
                vol = (df_ts.groupby('TRECHO_STD').size().reset_index(name='n'))
                reg_rank = []
                for reg in REGIOES_TRECHOS_STD.keys():
                    std = REGIOES_TRECHOS_STD[reg]
                    reg_rank.append((reg, int(vol[vol['TRECHO_STD'].isin(std)]['n'].sum())))
                top_regs = [r for r,_ in sorted(reg_rank, key=lambda x:x[1], reverse=True)[:5] if _>0]
                dplot = dr[dr['REGIÃO'].isin(top_regs)] if top_regs else dr
                figdr = line_fig(dplot, 'PERIODO', 'Diferença (%)', 'REGIÃO',
                                 f"{ag} – Diferença vs Melhor Concorrente por Região ({visao})",
                                 percent=True, cmap=build_blue_gray_map(dplot['REGIÃO'].unique()))
                st.plotly_chart(figdr, use_container_width=True)

            # ---------- 9.7 Comparativo de Preços vs. Melhor Concorrente – agências principais
            st.subheader("9.7 Comparativo de Preços vs. Melhor Concorrente – Agências Principais")
            def diff_vs_best_by_period(df_in, ag):
                d = add_period_column(df_in, visao, sd_ts, ed_ts)
                rows=[]
                for per, dfp in d.groupby('PERIODO'):
                    pm_ag = dfp.loc[dfp['Agência/Companhia']==ag,'Preço'].mean()
                    comp = dfp[~dfp['Agência/Companhia'].isin(['123MILHAS','MAXMILHAS'])]
                    if pd.isna(pm_ag) or comp.empty: continue
                    best = comp.groupby('Agência/Companhia')['Preço'].mean().min()
                    if pd.isna(best) or best==0: continue
                    rows.append({'PERIODO':per,'Agência':ag,'Diferença (%)':(pm_ag-best)/best*100})
                return pd.DataFrame(rows)

            lines=[]
            for ag in [a for a in ['123MILHAS','MAXMILHAS'] if a in df_ts['Agência/Companhia'].unique()]:
                d = diff_vs_best_by_period(df_ts, ag)
                if not d.empty: lines.append(d)
            if lines:
                dall = pd.concat(lines, ignore_index=True)
                cmapA = build_blue_gray_map(dall['Agência'].unique())
                figA = line_fig(dall, 'PERIODO', 'Diferença (%)', 'Agência',
                                f"Diferença vs Melhor Concorrente ({visao}) – Agências Principais",
                                percent=True, cmap=cmapA)
                st.plotly_chart(figA, use_container_width=True)
            else:
                st.info("Sem dados para comparativo por agência.")

            # ---------- 9.8 Quantidade de Ofertas por Ranking (123, MAX e Melhor Preço) – Hora e Data
            st.subheader("9.8 Quantidade de Ofertas por Ranking (123, MAX e Melhor Preço) – Hora e Data")
            dhr = df_ts.copy(); dhr['HORA'] = dhr['Data/Hora da Busca'].dt.floor('H')
            series=[]
            for ag in [a for a in ['123MILHAS','MAXMILHAS'] if a in dhr['Agência/Companhia'].unique()]:
                s = dhr.groupby(['HORA','RANKING','Agência/Companhia']).size().reset_index(name='Ofertas')
                s = s[s['Agência/Companhia']==ag]; s['Série']=ag; series.append(s)
            win = dhr[dhr['RANKING']==1].groupby(['HORA']).size().reset_index(name='Ofertas')
            win['RANKING']=1; win['Agência/Companhia']='*'; win['Série']='Melhor Preço'; series.append(win)
            if series:
                dfR = pd.concat(series, ignore_index=True)
                for rnk in [1,2,3]:
                    dplot = dfR[dfR['RANKING']==rnk]
                    if dplot.empty: continue
                    cmapR = build_blue_gray_map(dplot['Série'].unique())
                    fr = line_fig(dplot, 'HORA', 'Ofertas', 'Série',
                                  f"Ranking {rnk} por hora", percent=False, cmap=cmapR)
                    fr.update_yaxes(title_text="Qtd Ofertas")
                    st.plotly_chart(fr, use_container_width=True)

            # ---------- 9.9 % por Ranking – dentro da Agência (linha) – Hora e Data
            st.subheader("9.9 Participação (%) por Ranking – dentro da Agência (linha) – Hora e Data")
            dpa = dhr.copy()
            out=[]
            tot_mkt = dpa.groupby('HORA').size().reset_index(name='TOT')
            win_mkt = dpa[dpa['RANKING']==1].groupby('HORA').size().reset_index(name='WIN')
            m = tot_mkt.merge(win_mkt, on='HORA', how='left').fillna(0.0)
            m['Participação (%)']=np.where(m['TOT']>0, m['WIN']/m['TOT']*100, np.nan); m['Série']='Melhor Preço'
            out.append(m[['HORA','Participação (%)','Série']])
            for ag in [a for a in ['123MILHAS','MAXMILHAS'] if a in dpa['Agência/Companhia'].unique()]:
                tot = dpa[dpa['Agência/Companhia']==ag].groupby('HORA').size().reset_index(name='TOT')
                win = dpa[(dpa['Agência/Companhia']==ag)&(dpa['RANKING']==1)].groupby('HORA').size().reset_index(name='WIN')
                x = tot.merge(win, on='HORA', how='left').fillna(0.0); x['Participação (%)']=np.where(x['TOT']>0, x['WIN']/x['TOT']*100, np.nan)
                x['Série']=ag; out.append(x[['HORA','Participação (%)','Série']])
            dfp = pd.concat(out, ignore_index=True)
            cmapP = build_blue_gray_map(dfp['Série'].unique())
            fp = line_fig(dfp, 'HORA', 'Participação (%)', 'Série',
                          "% de Ranking 1 dentro da Agência (por hora)", percent=True, cmap=cmapP)
            st.plotly_chart(fp, use_container_width=True)

            # ---------- 9.10 % por Ranking – dentro do Ranking (coluna) – Hora e Data
            st.subheader("9.10 Participação (%) por Ranking – dentro do Ranking (coluna) – Hora e Data")
            dcol = dhr[dhr['RANKING']==1]
            if not dcol.empty:
                cnt = dcol.groupby(['HORA','Agência/Companhia']).size().reset_index(name='Q')
                tot = cnt.groupby('HORA')['Q'].sum().reset_index(name='TOT')
                share = cnt.merge(tot, on='HORA'); share['Participação (%)']=np.where(share['TOT']>0, share['Q']/share['TOT']*100, np.nan)
                share['Série'] = share['Agência/Companhia'].replace({'123MILHAS':'123MILHAS','MAXMILHAS':'MAXMILHAS'})
                share = share[share['Série'].isin(['123MILHAS','MAXMILHAS'])]
                if not share.empty:
                    cmapC = build_blue_gray_map(share['Série'].unique())
                    fc = line_fig(share, 'HORA', 'Participação (%)', 'Série',
                                  "Participação no Ranking 1 por hora (dentro do Ranking)", percent=True, cmap=cmapC)
                    st.plotly_chart(fc, use_container_width=True)
                else:
                    st.info("Sem dados para 123/MAX no Ranking 1.")
            else:
                st.info("Sem dados no Ranking 1 por hora.")

else:
    st.warning("Nenhum dado carregado.")


