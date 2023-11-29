import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import re
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas.io.sql as sqlio
import psycopg2
import pandas as pd
import sqlite3


psw='proyecto3'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Define los estilos CSS personalizados para mejorar la apariencia del dashboard
app.layout = html.Div([

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Panorama', value='tab-1'),
        dcc.Tab(label='Usa nuestra Herramienta', value='tab-2'),
    ]),
    html.Div(id='tab-content'),

], className='container')

########################
import psycopg2
import pandas as pd
import sqlite3
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import psycopg2

#Contraseña Conexión BD
psw='proyecto3'

def Grafica1():
    #CONECTARSE

    engine = psycopg2.connect(
    dbname="p3",
    user="postgres",
    password=psw,
    host="databasep3.cut51ddih3ok.us-east-1.rds.amazonaws.com",
    port='5432'
    )
    cursor = engine.cursor()
    
    ###########CONSULTA###########

    return dcc.Graph(figure=fig)

def Grafica2():
    # Establecer la conexión a la base de datos
    engine = psycopg2.connect(
    dbname="p3",
    user="postgres",
    password=psw,
    host="databasep3.cut51ddih3ok.us-east-1.rds.amazonaws.com",
    port='5432'
    )
    cursor = engine.cursor()

    ###########CONSULTA###########

    return dcc.Graph(figure=fig)

def Grafica3():
    #CONECTARSE
    engine = psycopg2.connect(
    dbname="p3",
    user="postgres",
    password=psw,
    host="databasep3.cut51ddih3ok.us-east-1.rds.amazonaws.com",
    port='5432'
    )

    cursor = engine.cursor()

    ###########CONSULTA###########


    return dcc.Graph(figure=fig)




#################

tab_1_content = html.Div([

html.Div([
    html.H1('Visualización Estudiantil en Magdalena', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
    html.P('A continuación encontrarás tres visualizaciones que te darán a conocer la situación actual de la Prueba Saber 11 de individuos del departamento de Magdalena', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),

    html.Br(),

    html.Div([
    Grafica1()
    ]),

    html.Br(),

    html.Div([
    Grafica2()
    ]),

    html.Br(),

    html.Div([
    Grafica3()
    ]),
    
    html.Br(),
    html.Br(),

    html.Div([
    html.P('¡En la pestaña "Usa nuestra Herramienta" te invitamos a calcular tu probabilidad de.......!', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ]),


    html.Br(),
    html.Br(),

], style={'backgroundColor': '#f2f2f2'})

tab_2_content = html.Div([
    html.Div([
        html.H1('Herramienta de Predicción del Saber 11', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
        html.P('¡Estas a un paso de pasar a la universidad! Esta herramienta utiliza tus datos socioeconómicos y del colegio para predecir la probabilidad de quedar en un rango de calificación en las 5 materias evaluadas en el icfes. Te invitamos a responder las siguientes preguntas:', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),
    
    html.Br(),

    html.Div([
        html.Div([
            html.Label('Periodo'),
            dcc.Dropdown(id='P', options=[{'label': '2019', 'value': 2019}, {'label': '2022', 'value': 2022}], placeholder='Selecciona el periodo cursado'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('Tipo de Documento'),
            dcc.Dropdown(id = 'Doc', options =[{'label':'TI', 'value': 1}, {'label':'CC', 'value': 2}, {'label':'CE', 'value': 3} , {'label':'CR', 'value': 4}, {'label':'PEP', 'value':5}, {'label':'NES', 'value': 6}, {'label':'PE', 'value':7}, {'label':'CCB', 'value':8}, {'label':'PPT', 'value':9}, {'label':'PC', 'value':10}], placeholder='Selecciona tu tipo de documento'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('Área de ubicacioón del colegio'),
            dcc.Dropdown(id='U', options=[{'label': 'Rural', 'value': 1}, {'label': 'Urbano', 'value': 2}], placeholder='Selecciona el área'),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('¿Tu colegio es/era bilingüe?'),
            dcc.Dropdown(id='Bil', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 2}], placeholder='¿Bilingüe?'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('¿Qué calendario es/era tu colegio?'),
            dcc.Dropdown(id='Cal', options=[{'label': 'A', 'value': 1}, {'label': 'B', 'value': 2}], placeholder='Calendario?'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
                html.Label('¿Qué tipo de educación brinda/brindaba su colegio?'),
                dcc.Dropdown(id='Edu', options=[{'label':'TÉCNICO/ACADÉMICO', 'value':1},{'label':'TÉCNICO', 'value':2}, {'label':'ACADÉMICO', 'value':3}, {'label':'NO APLICA', 'value':4}], placeholder='Educación del colegio'),
            ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),


    html.Div([
        html.Div([
            html.Label('¿En qué municipio que ubicado el colegio?'),
            dcc.Dropdown(id='Mun', options=[{'label': 'SABANAS DE SAN ÁNGEL', 'value': 1}, {'label': 'ZAPAYÁN', 'value': 2 },{'label': 'SANTA MARTA', 'value': 3},{'label': 'PUEBLOVIEJO', 'value': 4},{'label': 'PEDRAZA', 'value': 5}, {'label': 'SAN SEBASTIÁN DE BUENAVISTA', 'value': 6},{'label': 'SANTA ANA', 'value': 7},{'label': 'PIVIJAY', 'value': 8},{'label': 'FUNDACIÓN', 'value': 9},{'label': 'ZONA BANANERA', 'value':10},{'label': 'EL RETÉN', 'value':11},{'label': 'PLATO', 'value':12},{'label': 'EL BANCO', 'value':13},{'label': 'SAN ZENÓN', 'value':14},{'label': 'GUAMAL', 'value':15},{'label': 'CHIVOLO', 'value':16}, {'label': 'ARIGUANÍ', 'value':17}, {'label': 'SALAMINA', 'value':18}, {'label': 'CIÉNAGA', 'value':19}, {'label': 'CERRO DE SAN ANTONIO', 'value':20}, {'label': 'SITIONUEVO', 'value':21}, {'label': 'ALGARROBO', 'value':22}, {'label': 'ARACATACA', 'value':23}, {'label': 'SANTA BÁRBARA DE PINTO', 'value':24}, {'label': 'REMOLINO', 'value':25}, {'label': 'NUEVA GRANADA', 'value':26}, {'label': 'PIJIÑO DEL CARMEN', 'value':27}, {'label': 'CONCORDIA', 'value':28}, {'label': 'TENERIFE', 'value':29}, {'label': 'EL PIÑÓN', 'value':30}], placeholder='Seleccione el municipio'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿Qué jornada escolar tiene su colegio?'),
            dcc.Dropdown(id='Jornada', options=[{'label':'TARDE', 'value':1},{'label':'MAÑANA', 'value':2}, {'label':'NOCHE', 'value':3}, {'label':'COMPLETA', 'value':4}, {'label':'SABATINA', 'value':5}, {'label':'UNICA', 'value':6}], placeholder='Jornada del colegio'),
        ], className='four columns', style={'marginTop': '10px'}),

         html.Div([
            html.Label('¿Qué naturaleza tiene su colegio?'),
            dcc.Dropdown(id='Nat', options=[{'label':'OFICIAL', 'value':1},{'label':'NO OFICIAL', 'value':2}], placeholder='Naturaleza del colegio'),
        ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('Seleccione su género'),
            dcc.Dropdown(id='Genero', options=[{'label': 'M', 'value': 1}, {'label': 'F', 'value': 2}], placeholder='Género'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('Seleccione la educación de su madre'),
            dcc.Dropdown(id = 'EduMadre', options =[{'label':'Educación profesional completa', 'value': 1}, {'label':'Secundaria (Bachillerato) completa', 'value': 2}, {'label':'Primaria incompleta', 'value': 3} , {'label':'Primaria completa', 'value': 4}, {'label':'Secundaria (Bachillerato) incompleta', 'value':5}, {'label':'Técnica o tecnológica incompleta', 'value': 6}, {'label':'Técnica o tecnológica completa', 'value':7}, {'label':'No sabe', 'value':8}, {'label':'Postgrado', 'value':9}, {'label':'Ninguno', 'value':10}, {'label':'Educación profesional incompleta', 'value':11}, {'label':'No Aplica', 'value':12}], placeholder='Educación Madre'),
        ], className='four columns', style={'marginTop': '10px'}),

         html.Div([
            html.Label('Seleccione la educación de su padre'),
            dcc.Dropdown(id = 'EduPadre', options =[{'label':'Educación profesional completa', 'value': 1}, {'label':'Secundaria (Bachillerato) completa', 'value': 2}, {'label':'Primaria incompleta', 'value': 3} , {'label':'Primaria completa', 'value': 4}, {'label':'Secundaria (Bachillerato) incompleta', 'value':5}, {'label':'Técnica o tecnológica incompleta', 'value': 6}, {'label':'Técnica o tecnológica completa', 'value':7}, {'label':'No sabe', 'value':8}, {'label':'Postgrado', 'value':9}, {'label':'Ninguno', 'value':10}, {'label':'Educación profesional incompleta', 'value':11}, {'label':'No Aplica', 'value':12}], placeholder='Educación Padre'),
        ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('Seleccione el estrato de su vivienda'),
            dcc.Dropdown(id='Estrato', options=[{'label': 'Estrato 1', 'value': 1}, {'label': 'Estrato 2', 'value': 2}, {'label': 'Estrato 3', 'value': 3}, {'label': 'Estrato 4', 'value': 4}, {'label': 'Estrato 5', 'value': 5}, {'label': 'Estrato 6', 'value': 6}, {'label': 'Sin Estrato', 'value': 7}], placeholder='Estrato Vivienda'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿Cuantas personas habitan en su vivienda?'),
            dcc.Dropdown(id = 'PersonasViv', options =[{'label':'1 a 2', 'value': 1}, {'label':'3 a 4', 'value': 2}, {'label':'5 a 6', 'value': 3} , {'label':'7 a 8', 'value': 4}, {'label':'9 o más', 'value':5}], placeholder='Habitantes Vivienda'),
        ], className='four columns', style={'marginTop': '10px'}),

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

         html.Div([
            html.Label('¿En su vivienda cuentan con lavadora?'),
            dcc.Dropdown(id = 'Lavadora', options =[{'label':'No', 'value': 0}, {'label':'Si', 'value': 1}], placeholder='¿Lavadora?'),
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
], style={'backgroundColor': '#f2f2f2'})



def pred_ingles(P, C, U, Car, N, F, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_ingles.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'punt_ingles'), ('cole_area_ubicacion', 'fami_tienecomputador'), ('cole_area_ubicacion', 'periodo'), ('cole_area_ubicacion', 'fami_personashogar'), ('cole_calendario', 'periodo'), ('cole_calendario', 'fami_tieneautomovil'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_area_ubicacion'), ('cole_caracter', 'punt_ingles'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'punt_ingles'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_naturaleza', 'punt_ingles'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_personashogar'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'cole_calendario'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'periodo'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tieneinternet', 'fami_personashogar'), ('punt_ingles', 'estu_tipodocumento'), ('punt_ingles', 'periodo'), ('punt_ingles', 'fami_tienecomputador'), ('punt_ingles', 'fami_personashogar'), ('punt_ingles', 'cole_calendario')])

    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['punt_ingles'], evidence={'periodo': P, 'cole_calendario': C, 'cole_area_ubicacion': U, 'cole_caracter': Car, 'cole_naturaleza': N, 'fami_personashogar': F, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    return resp

@app.callback(Output('tab-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab_1_content
    elif tab == 'tab-2':
        return tab_2_content

@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('P', 'value'),
    State('Doc', 'value'),
    State('U', 'value'),
    State('Bil', 'value'),
    State('Cal', 'value'),
    State('Edu', 'value'),
    State('Mun', 'value'),
    State('Jornada', 'value'),
    State('Nat', 'value'),
    State('Genero', 'value'),
    State('EduMadre', 'value'),
    State('EduPadre', 'value'),
    State('Estrato', 'value'),
    State('PersonasViv', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value'),
    State('Lavadora', 'value')]
)


################LECTURA######################

def pred_lectura(P, U, Car, N, F, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_lectura.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('periodo', 'cole_caracter'), ('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'fami_tienecomputador'), ('cole_area_ubicacion', 'punt_lectura_critica'), ('cole_area_ubicacion', 'fami_personashogar'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_area_ubicacion'), ('cole_caracter', 'punt_lectura_critica'), ('cole_caracter', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'punt_lectura_critica'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'punt_lectura_critica'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_personashogar'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_tieneautomovil', 'fami_tienecomputador'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tieneinternet', 'fami_personashogar'), ('punt_lectura_critica', 'estu_tipodocumento'), ('punt_lectura_critica', 'fami_personashogar')])
    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['punt_lectura_critica'], evidence={'periodo': P, 'cole_area_ubicacion': U, 'cole_caracter': Car, 'cole_naturaleza': N, 'fami_personashogar': F, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    return resp

@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('P', 'value'),
    State('Doc', 'value'),
    State('U', 'value'),
    State('Bil', 'value'),
    State('Cal', 'value'),
    State('Edu', 'value'),
    State('Mun', 'value'),
    State('Jornada', 'value'),
    State('Nat', 'value'),
    State('Genero', 'value'),
    State('EduMadre', 'value'),
    State('EduPadre', 'value'),
    State('Estrato', 'value'),
    State('PersonasViv', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value'),
    State('Lavadora', 'value')]
)


##################MATEMATICAS####################

def pred_mate(P, U, N, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_mate.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('periodo', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_mcpio_ubicacion', 'punt_matematicas'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'punt_matematicas'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneinternet'), ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'fami_tienecomputador'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tieneinternet', 'cole_area_ubicacion'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('punt_matematicas', 'estu_tipodocumento'), ('punt_matematicas', 'fami_tieneautomovil'), ('punt_matematicas', 'fami_tienecomputador')])
    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['punt_matematicas'], evidence={'periodo': P, 'cole_area_ubicacion': U, 'cole_naturaleza': N, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    return resp

@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('P', 'value'),
    State('Doc', 'value'),
    State('U', 'value'),
    State('Bil', 'value'),
    State('Cal', 'value'),
    State('Edu', 'value'),
    State('Mun', 'value'),
    State('Jornada', 'value'),
    State('Nat', 'value'),
    State('Genero', 'value'),
    State('EduMadre', 'value'),
    State('EduPadre', 'value'),
    State('Estrato', 'value'),
    State('PersonasViv', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value'),
    State('Lavadora', 'value')]
)


################NATURALES##############

def pred_natu(P, U, N, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_natural.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('estu_tipodocumento', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_mcpio_ubicacion', 'punt_c_naturales'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'punt_c_naturales'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneinternet'), ('cole_naturaleza', 'periodo'), ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'fami_tienecomputador'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tienecomputador', 'periodo'), ('fami_tieneinternet', 'cole_area_ubicacion'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('punt_c_naturales', 'estu_tipodocumento'), ('punt_c_naturales', 'periodo'), ('punt_c_naturales', 'fami_tieneautomovil'), ('punt_c_naturales', 'fami_tienecomputador')])
    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['punt_c_naturales'], evidence={'periodo': P, 'cole_area_ubicacion': U, 'cole_naturaleza': N, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    return resp

@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('P', 'value'),
    State('Doc', 'value'),
    State('U', 'value'),
    State('Bil', 'value'),
    State('Cal', 'value'),
    State('Edu', 'value'),
    State('Mun', 'value'),
    State('Jornada', 'value'),
    State('Nat', 'value'),
    State('Genero', 'value'),
    State('EduMadre', 'value'),
    State('EduPadre', 'value'),
    State('Estrato', 'value'),
    State('PersonasViv', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value'),
    State('Lavadora', 'value')]
)


###################SOCIALES###################

def pred_sociales(P, U, N, Mun, Comp, Auto, Int, Doc):
    datos = 'BD_sociales.csv'
    df = pd.read_csv(datos)
    
    modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('periodo', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_mcpio_ubicacion', 'punt_sociales_ciudadanas'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'punt_sociales_ciudadanas'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneinternet'), ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'periodo'), ('cole_naturaleza', 'fami_tienecomputador'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tienecomputador', 'periodo'), ('fami_tieneinternet', 'cole_area_ubicacion'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('punt_sociales_ciudadanas', 'estu_tipodocumento'), ('punt_sociales_ciudadanas', 'periodo'), ('punt_sociales_ciudadanas', 'fami_tienecomputador')])
    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['punt_sociales_ciudadanas'], evidence={'periodo': P, 'cole_area_ubicacion': U, 'cole_naturaleza': N, 'cole_mcpio_ubicacion': Mun, 'fami_tienecomputador': Comp, 'fami_tieneautomovil':Auto, 'fami_tieneinternet': Int, 'estu_tipodocumento': Doc})

    return resp

@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('P', 'value'),
    State('Doc', 'value'),
    State('U', 'value'),
    State('Bil', 'value'),
    State('Cal', 'value'),
    State('Edu', 'value'),
    State('Mun', 'value'),
    State('Jornada', 'value'),
    State('Nat', 'value'),
    State('Genero', 'value'),
    State('EduMadre', 'value'),
    State('EduPadre', 'value'),
    State('Estrato', 'value'),
    State('PersonasViv', 'value'),
    State('Auto', 'value'),
    State('Compu', 'value'),
    State('Internet', 'value'),
    State('Lavadora', 'value')]
)



###############################################

def update_output(n_clicks, MS, G, AE, PQ, AG, D, C, AO):
    try:
        if n_clicks >= 0:
        
            lim_Edad = [16, 30, 45, 63]
            labels_Edad = [1, 2, 3]
            AE = pd.cut([AE], bins=lim_Edad, labels=labels_Edad)[0]

            lim_Notas = [-1, 49, 99, 149, 201]
            labels_Notas = [1, 2, 3, 4]
            PQ = pd.cut([PQ], bins=lim_Notas, labels=labels_Notas)[0]
            AG = pd.cut([AG], bins=lim_Notas, labels=labels_Notas)[0]

            probabilidad = by_pred(MS, AO, C, PQ, AG, D, G, AE)
            
            # Crea un gráfico de barras para visualizar la probabilidad
            df = pd.DataFrame({'Probabilidad': [probabilidad], 'Resultado': ['Probabilidad de Graduación']})

            prob_df = pd.DataFrame({'Categoría': ['Graduación', 'Retiro'], 'Probabilidad': [probabilidad.values[1], probabilidad.values[0]]})

            #fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=600,
             #        labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
              #       color='Categoría', title='Probabilidad de Graduación vs. Probabilidad de Retiro')

            colors = {'Graduación': 'green', 'Retiro': 'lightgreen'}

            fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=600,
                        labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
                        color_discrete_map=colors,  # Establecer los colores
                        title='Probabilidad de Graduación vs. Probabilidad de Retiro')
            
            # Personaliza el diseño del gráfico
            fig.update_traces(marker_color=['green', 'lightgreen'])
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(legend_title_text='Resultado')
            #fig = {'data': [{'x': ['Probabilidad de Graduación'], 'y': [probabilidad.values[1]], 'type': 'bar'}],'layout': {'xaxis': {'title': 'Resultado'}, 'yaxis': {'title': 'Probabilidad'}}}
            #fig = px.bar(df, x='Resultado', y='Probabilidad', text='Probabilidad', height=400)

            resp = probabilidad.values[1]*100
            
            return f'Probabilidad de Graduación: {round(resp, 2)}%', fig
        
    except Exception as e:
        return f'Error: {str(e)}', {}

#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True,port =8070)
        

