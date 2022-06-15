import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import plotly.express as px
from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go

conn = sqlite3.connect("f1.db", check_same_thread=False)

circuits = pd.read_csv("circuits.csv")
constructor_results = pd.read_csv("constructor_results.csv")
constructor_standings = pd.read_csv("constructor_standings.csv")
constructors = pd.read_csv("constructors.csv")
driver_standings = pd.read_csv("driver_standings.csv")
drivers = pd.read_csv("drivers.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
sprint_results = pd.read_csv("sprint_results.csv")

circuits.to_sql("circuits", conn)
constructor_results.to_sql("constructor_results", conn)
constructor_standings.to_sql("constructor_standings", conn)
constructors.to_sql("constructors", conn)
driver_standings.to_sql("driver_standings", conn)
drivers.to_sql("drivers", conn)
races.to_sql("races", conn)
results.to_sql("results", conn)
sprint_results.to_sql("sprint_results", conn)

app = JupyterDash(__name__, external_stylesheets=['main.css'])

chosen_year = dcc.Input(type='number', min=1950, max=2022, step=1, value='2019')

app.layout = html.Div(children=[
    html.H1(children='Formula 1 Results 1950 - 2022',
           style = {
               'textAlign':'center',
               'color':'#FFFFFF',
               'backgroundColor':'black',
               'font-family':'Arial, Helvetica, sans-serif'
           }),

    chosen_year,

    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Driver Standings Trace', value='tab-1-example-graph', style={'font-family':'Arial, Helvetica, sans-serif','color':'white','backgroundColor':'black'}),
        dcc.Tab(label='Driver Standings Table', value='tab-2-example-graph',style={'font-family':'Arial, Helvetica, sans-serif','color':'white','backgroundColor':'black'}),
        dcc.Tab(label='Constructor Standings Trace', value='tab-3-example-graph',style={'font-family':'Arial, Helvetica, sans-serif','color':'white','backgroundColor':'black'}),
        dcc.Tab(label='Constructor Standings Table', value='tab-4-example-graph',style={'font-family':'Arial, Helvetica, sans-serif','color':'white','backgroundColor':'black'})
    ]),
    html.Div(id='tabs-content-example-graph')
], style={'color':'white','backgroundColor':'black'})

@app.callback(
    dash.Output('tabs-content-example-graph', 'children'),
    dash.Input('tabs-example-graph', 'value'),
    dash.Input(component_id = chosen_year, component_property='value')
)
def render_content(tab, selected_year):
    standing_query = 'SELECT code as Driver, surname, points, round FROM driver_standings INNER JOIN drivers ON drivers.driverId = driver_standings.driverId INNER JOIN races ON driver_standings.raceId = races.raceId WHERE year = ' + str(selected_year) + ' ORDER BY round;'
    standing_chart_df = pd.read_sql(standing_query, conn)
    
    standing_query2 = 'SELECT position, forename, surname, number, nationality, points, wins FROM driver_standings INNER JOIN drivers ON drivers.driverId = driver_standings.driverId WHERE raceId = (SELECT MAX(raceId) FROM races WHERE year = ' + str(selected_year) + ') ORDER BY position;'
    standing_chart_df2 = pd.read_sql(standing_query2, conn)
        
    constr_query1 = 'SELECT constructors.name as Constructor, points, round FROM constructor_standings INNER JOIN constructors ON constructors.constructorId = constructor_standings.constructorId INNER JOIN races ON constructor_standings.raceId = races.raceId WHERE year = ' + str(selected_year) + ' ORDER BY round;'
    constr_df1 = pd.read_sql(constr_query1, conn)
    
    constr_query2 = 'SELECT position, constructors.name as Constructor, points, wins FROM constructor_standings INNER JOIN constructors ON constructors.constructorId = constructor_standings.constructorId WHERE raceId = (SELECT MAX(raceId) FROM races WHERE year = ' + str(selected_year) + ') ORDER BY position;'
    constr_df2 = pd.read_sql(constr_query2, conn)

    
    if tab == 'tab-1-example-graph':
        if (selected_year > 2004):
            return html.Div([
                dcc.Graph(
                    figure=px.line(standing_chart_df, x='round',y='points', color='Driver', markers=True, template='plotly_dark')

                )
            ])
        else:
            return html.Div([
                dcc.Graph(
                    figure=px.line(standing_chart_df, x='round',y='points', color='surname', markers=True, template='plotly_dark')
                )
            ])
    elif tab == 'tab-2-example-graph':
        return html.Div([
            dcc.Graph(
                id='graph-2-tabs-dcc',
                figure=go.Figure(data=[go.Table(
    header=dict(values=list(standing_chart_df2.columns),
                fill_color='ghostwhite',
                line_color='darkslategray',
                align='left'),
    cells=dict(values=standing_chart_df2.transpose().values.tolist(),
               line_color='darkslategray',
               fill_color='ghostwhite',
               align='left'))
            ])               
        )
    ])
    
    elif tab == 'tab-3-example-graph':
        return html.Div([
                dcc.Graph(
                    figure=px.line(constr_df1, x='round',y='points', color='Constructor', markers=True, template='plotly_dark')
                )
            ])
    
    elif tab == 'tab-4-example-graph':
        return html.Div([
            dcc.Graph(
                id='graph-2-tabs-dcc',
                figure=go.Figure(data=[go.Table(
    header=dict(values=list(constr_df2.columns),
                fill_color='ghostwhite',
                line_color='darkslategray',
                align='left'),
    cells=dict(values=constr_df2.transpose().values.tolist(),
               line_color='darkslategray',
               fill_color='ghostwhite',
               align='left'))
            ])               
        )
    ])
    
if __name__ == '__main__':
    app.run_server()