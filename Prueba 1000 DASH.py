import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.inference import VariableElimination
import pandas.io.sql as sqlio
import psycopg2
import sqlite3
import psycopg2
import pandas as pd
import sqlite3
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import psycopg2
import plotly.graph_objects as go


# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Diseño de la aplicación
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Pestaña 1', value='tab1'),
        dcc.Tab(label='Pestaña 2', value='tab2'),
    ]),
    html.Div(id='content')
], className='container')


def Grafica1():
    #CONECTARSE
    import psycopg2
    import pandas as pd
    import sqlite3

    engine = psycopg2.connect(
        dbname="world",
        user="postgres",
        password="navidadp3",  # Asegúrate de definir la variable 'psw' antes de usarla
        host="p3.curxufagptbe.us-east-1.rds.amazonaws.com",
        port='5432'
    )
    cursor = engine.cursor()
    
    ###########CONSULTA###########
    query = """
    SELECT periodo, 
       punt_global AS "Cuartil",  -- Cambiar el nombre de la columna
       COUNT(*) AS cantidad
       FROM bd_q
       WHERE periodo IN (2019, 2022)
       GROUP BY periodo, punt_global;
    """
    
    # Ejecutar la consulta y obtener el resultado como un DataFrame
    df = pd.read_sql_query(query, engine)

    # Mapear los valores de punt_global según la asignación dada
    mapping = {
        1: 'Cuartil 1',
        2: 'Cuartil 2',
        3: 'Cuartil 3',
        4: 'Cuartil 4'
    }
    
    # Aplicar el reemplazo
    df['Cuartil'] = df['Cuartil'].replace(mapping)

    # Filtrar solo los años que te interesan
    df = df[df['periodo'].isin([2019, 2022])]
    
    # Calcular la proporción de cada categoría en relación con la cantidad total por año
    df['proporcion'] = df.groupby('periodo')['cantidad'].transform(lambda x: x / x.sum())

    # Crear la figura de la gráfica de barras
    fig = px.bar(
        df, 
        x='periodo', 
        y='proporcion',
        color='Cuartil',
        labels={'proporcion': 'Porcentaje', 'periodo': 'Año'},
        title='Porcentaje de notas ICFES por cuartiles antes y después de pandemia en Magdalena',
        category_orders={'Cuartil': ['Cuartil 1', 'Cuartil 2', 'Cuartil 3', 'Cuartil 4']},
        color_discrete_sequence=px.colors.sequential.Greens_r,
        text=df['proporcion']
    )  
    
    # Personalizar el diseño de la gráfica
    fig.update_xaxes(title_text='', showticklabels=True, categoryorder='array', categoryarray=[2019,2022])  # Ocultar etiquetas y valores del eje x
    fig.update_yaxes(showgrid=False, showline=False, zeroline=False, title_text='', showticklabels=False)  # Ocultar etiquetas y valores del eje y

    # Personalizar el formato del texto y centrarlo
    fig.update_traces(texttemplate='%{text:.2%}', 
                      textfont=dict(color='white', size=18, family='Arial, sans-serif'), 
                      insidetextanchor='start',
                      textposition='inside')



    return dcc.Graph(figure=fig)


# Definir el contenido de las pestañas
tab1_layout = html.Div([
    html.Div([
    html.H1('Graduación y Deserción Estudiantil: Un Análisis Visual', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
    html.P('A continuación encontrarás tres visualizaciones que te darán a conocer la situación actual de graduación y deserción en tu universidad', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),

    html.Br(),

    html.Div([
    Grafica1()
    ]),

    html.Br(),

    html.Div([
    html.H1('Aqui va otra gráfica')
    ]),

    html.Br(),
    
    html.Div([
        html.H1('Aqui va otra gráfica')
        ]),

    html.Br(),
    html.Br(),

    html.Div([
    html.P('¡En la pestaña "Usa nuestra Herramienta" te invitamos a calcular tu probabilidad de graduación!', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ]),


    html.Br(),
    html.Br(),
])

tab2_layout = html.Div([
    html.Div([
        html.H1('Herramienta de Predicción del Saber 11', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
        html.P('¡Estas a un paso de pasar a la universidad! Esta herramienta utiliza tus datos socioeconómicos y del colegio para predecir la probabilidad de quedar en un rango de calificación en las 5 materias evaluadas en el icfes. Te invitamos a responder las siguientes preguntas:', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),
    
    html.Br(),

    html.Div([
        html.Div([
            html.Label('Tipo de Documento'),
            dcc.Dropdown(id = 'Doc', options =[{'label':'TI', 'value': 1}, {'label':'CC', 'value': 2}, {'label':'CE', 'value': 3} , {'label':'CR', 'value': 4}, {'label':'PEP', 'value':5}, {'label':'NES', 'value': 6}, {'label':'PE', 'value':7}, {'label':'CCB', 'value':8}, {'label':'PPT', 'value':9}, {'label':'PC', 'value':10}], placeholder='Selecciona tu tipo de documento'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('Área de ubicación del colegio'),
            dcc.Dropdown(id='U', options=[{'label': 'Rural', 'value': 1}, {'label': 'Urbano', 'value': 2}], placeholder='Selecciona el área'),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),
    html.Br(),


    html.Div([
        html.Div([
            html.Label('¿En qué municipio que ubicado el colegio?'),
            dcc.Dropdown(id='Mun', options=[{'label': 'SABANAS DE SAN ÁNGEL', 'value': 1}, {'label': 'ZAPAYÁN', 'value': 2 },{'label': 'SANTA MARTA', 'value': 3},{'label': 'PUEBLOVIEJO', 'value': 4},{'label': 'PEDRAZA', 'value': 5}, {'label': 'SAN SEBASTIÁN DE BUENAVISTA', 'value': 6},{'label': 'SANTA ANA', 'value': 7},{'label': 'PIVIJAY', 'value': 8},{'label': 'FUNDACIÓN', 'value': 9},{'label': 'ZONA BANANERA', 'value':10},{'label': 'EL RETÉN', 'value':11},{'label': 'PLATO', 'value':12},{'label': 'EL BANCO', 'value':13},{'label': 'SAN ZENÓN', 'value':14},{'label': 'GUAMAL', 'value':15},{'label': 'CHIVOLO', 'value':16}, {'label': 'ARIGUANÍ', 'value':17}, {'label': 'SALAMINA', 'value':18}, {'label': 'CIÉNAGA', 'value':19}, {'label': 'CERRO DE SAN ANTONIO', 'value':20}, {'label': 'SITIONUEVO', 'value':21}, {'label': 'ALGARROBO', 'value':22}, {'label': 'ARACATACA', 'value':23}, {'label': 'SANTA BÁRBARA DE PINTO', 'value':24}, {'label': 'REMOLINO', 'value':25}, {'label': 'NUEVA GRANADA', 'value':26}, {'label': 'PIJIÑO DEL CARMEN', 'value':27}, {'label': 'CONCORDIA', 'value':28}, {'label': 'TENERIFE', 'value':29}, {'label': 'EL PIÑÓN', 'value':30}], placeholder='Seleccione el municipio'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿Qué tipo de educación brinda/brindaba su colegio?'),
            dcc.Dropdown(id='Edu', options=[{'label':'TÉCNICO/ACADÉMICO', 'value':1},{'label':'TÉCNICO', 'value':2}, {'label':'ACADÉMICO', 'value':3}, {'label':'NO APLICA', 'value':4}], placeholder='Educación del colegio'),
        ], className='four columns', style={'marginTop': '10px'}),

         html.Div([
            html.Label('¿Qué naturaleza tiene su colegio?'),
            dcc.Dropdown(id='Nat', options=[{'label':'OFICIAL', 'value':1},{'label':'NO OFICIAL', 'value':2}], placeholder='Naturaleza del colegio'),
        ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),
    html.Br(),

    
    html.Div([
         html.Div([
            html.Label('¿En su vivienda cuentan con automóvil?'),
            dcc.Dropdown(id = 'Auto', options =[{'label':'No', 'value': 0}, {'label':'Si', 'value': 1}], placeholder='¿Auto?'),
        ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('¿En su vivienda cuentan con computadora?'),
            dcc.Dropdown(id = 'Compu', options =[{'label':'No', 'value': 0}, {'label':'Si', 'value': 1}], placeholder='¿Compu?'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿En su vivienda cuentan con internet?'),
            dcc.Dropdown(id = 'Internet', options =[{'label':'No', 'value': 0}, {'label':'Si', 'value': 1}], placeholder='Internet?'),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),

    ##

    html.Br(),

    html.Button('CALCULA TU RESULTADO', id='submit', n_clicks=0, style={'backgroundColor': '#6C9B37', 'color': 'white', 'fontSize': '18px'}),
    
    html.Br(),
    html.Br(),

    html.Div(id='output'),

    html.Br(),

    #html.Div[(id='probability-plot')]
    dcc.Graph(id='probability-plot'),  # Visualización de probabilidad de graduación o retiro
])

# Callback para actualizar el contenido basado en la pestaña seleccionada
@app.callback(Output('content', 'children'),
              [Input('tabs', 'value')])
def update_content(selected_tab):
    if selected_tab == 'tab1':
        return tab1_layout
    elif selected_tab == 'tab2':
        return tab2_layout
    else:
        return html.Div([])


def pred_global(U, Car, N, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_Cuartiles.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'punt_global'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_naturaleza', 'fami_tieneautomovil'), ('fami_tienecomputador', 'fami_tieneinternet'), ('punt_global', 'cole_naturaleza'), ('punt_global', 'estu_tipodocumento')])

    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)
    resp = infer.query(['punt_global'], evidence={'cole_area_ubicacion': U, 'cole_caracter': Car, 'cole_naturaleza': N, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    #respuesta = 'La probabilidad que obtengas un rango sobresaliente es de ' + str(round(resp.values[1], 4)*100) + '%'

    prob_df = pd.DataFrame({'Resultado': ['Demás', 'Sobresaliente'],'Probabilidad': [resp.values[0], resp.values[1]]})
    
    respuesta = prob_df

    return respuesta


@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('Doc', 'value'),
    State('U', 'value'),
    State('Mun', 'value'),
    State('Edu', 'value'),
    State('Nat', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value')]
)


def update_output(n_clicks, U, Car, N, Mun, Comp, Auto, Int, Doc):
    try:
        if n_clicks >= 0:
            probabilidad = pred_global(U, Car, N, Mun, Comp, Auto, Int, Doc)
            
            # Convertir las probabilidades a porcentajes
            prob_sobresaliente = round(probabilidad.iloc[1]['Probabilidad'] * 100, 2)
            prob_otros = 100 - prob_sobresaliente  # La probabilidad de "Otros" es el complemento de Sobresaliente
            
            # Crea un gráfico de donut para visualizar la probabilidad
            fig = go.Figure()

            # Agregar un trace de tipo 'pie' (donut chart)
            fig.add_trace(go.Pie(
                labels=['Sobresaliente', 'Otros'],
                values=[prob_sobresaliente, prob_otros],
                hole=0.4,  # Este valor controla el tamaño del agujero (donut hole)
                marker=dict(colors=['green', 'lightgreen']),
                textinfo='label+percent'
            ))

            # Actualizar el diseño del gráfico
            fig.update_layout(
                title='Probabilidad de sacar un excelente puntaje en el Icfes',
                annotations=[dict(text=f'{prob_sobresaliente}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
            )

            return f'Probabilidad de estar en el rango Sobresaliente: {prob_sobresaliente}%', fig
        
    except Exception as e:
        return f'Error: {str(e)}', {}




# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)