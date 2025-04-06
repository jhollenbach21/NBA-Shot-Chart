import dash
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, Input, Output, callback
from PIL import Image as PILImage
from sqlalchemy import create_engine
import os

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def heatmap_by_season(season, scale, engine):
    sqlquery = "SELECT loc_x, loc_y, shot_made FROM basketball_shots WHERE season = '"+ season + "'"
    df_season = pd.read_sql(sqlquery, engine)
    print(f"Data for graph: {df_season.head()}")
    num_boxes = 25
    box_dist = 50/num_boxes
    total_shots = 0
    total_makes = 0
    hoop_location = (0, 4)
    shot_matrix = np.empty([num_boxes,num_boxes])
    make_matrix = np.empty([num_boxes, num_boxes], dtype=float)
    for j in range(num_boxes):
        for i in range(num_boxes):
            total_shots_at_spot = len(df_season[df_season['loc_x'].between(-25+i*box_dist, -25+(i+1)*box_dist) &
                                 df_season['loc_y'].between(j*box_dist,(j+1)*box_dist)])
            current_block = df_season[df_season['loc_x'].between(-25 + i * box_dist, -25 + (i + 1) * box_dist) &
                                                df_season['loc_y'].between(j * box_dist, (j + 1) * box_dist)]
            total_makes_at_spot = len(current_block[current_block['shot_made'] == True])
            shot_matrix[j][i]= total_shots_at_spot+1
            if(total_shots_at_spot == 0):
                make_matrix[j][i] = 0
            else:
                make_matrix[j][i] = (total_makes_at_spot / total_shots_at_spot)
            total_shots+=total_shots_at_spot
    shot_matrix = shot_matrix/total_shots
    text_matrix = shot_matrix
    if(scale == 'Sqrt'):
        shot_matrix = np.sqrt(shot_matrix)
    elif(scale == 'Log'):
        shot_matrix = np.log(shot_matrix)

    hover_text = np.array([["Field Goal %: {:.2%}<br>% of Total Shots: {:.2%}<br>Distance from hoop (ft): {:.2f}".format(
                            make_matrix[j][i], text_matrix[j][i], 
                            distance(-25 + i * box_dist + box_dist / 2, j * box_dist + box_dist / 2, hoop_location[0], hoop_location[1])) for i in range(num_boxes)] for j in range(num_boxes)])
    
    if(scale != 'Log' and scale != 'Sqrt'):
        scale_name = ''
    else:
        scale_name = scale

    heatmap = go.Heatmap(z=shot_matrix, text=hover_text, hoverinfo='text', colorbar={"title": scale_name + ' Shot Frequency', "orientation" :'h'})

    fig = go.Figure(data=[heatmap])
    
    court_image = PILImage.open("basketball_court.png")

    fig.update_layout(
    title={
        'text': season + " NBA Shot Locations",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        images=[
        go.layout.Image(
            source=court_image,
            xref="x",
            yref="y",
            x=-10.64,
            y=25.65,
            sizex=46,
            sizey=29,
            sizing="stretch",
            opacity=.1,
            layer="above"
        )
    ],
    autosize=False,
    width=600,
    height=600)
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig

def find_marginal(data, l_bound, u_bound, horizontal = True):
    data = data.sort_values()
    support = np.linspace(l_bound,u_bound,100)
    p_x = np.zeros_like(support)
    r = (u_bound-l_bound)/15
    for i,x in enumerate(support):
        p_x[i] = sum(scipy.stats.norm.pdf(data,x,r)) / len(data)
    if(horizontal):
        df = pd.DataFrame({
        'x': support,
        'p_x': p_x,})
        return df
    else:
        df = pd.DataFrame({
        'p_y': p_x,
        'y': support})
        return df

def distance_kde(season, engine):
    sqlquery = "SELECT loc_x, loc_y FROM basketball_shots WHERE season = '"+ season + "'"
    df_season = pd.read_sql(sqlquery, engine)
    hoop_location = (0, 4)
    shot_distances = pd.Series([
        distance(df_season['loc_x'][i], df_season['loc_y'][i], hoop_location[0], hoop_location[1]) 
        for i in range(len(df_season))
    ])
    marg_dist = find_marginal(shot_distances, 0, 35)
    fig = px.scatter(marg_dist, 'x', 'p_x')
    fig.data[0].update(mode='lines')
    fig.add_vline(x=22.5, line_width=1, line_dash="dash", line_color="black", annotation_text ='corner 3', annotation_position = 'bottom left')
    fig.add_vline(x=23.75, line_width=1, line_dash="dash", line_color="black", annotation_text ='anywhere else 3', annotation_position = 'top right')
    fig.update_layout(
    title={
        'text': season + " NBA Shot Probability Based on Distance",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title = 'Shot Distance',
    yaxis_title = 'Probabillity of Shot')
    return fig

def heatmap_by_player(player_name, scale, engine):
    sql_query = "SELECT loc_x, loc_y, shot_made FROM basketball_shots WHERE player_name = '"+ player_name + "'"
    df_player = pd.read_sql(sql_query, engine)
    num_boxes = 25
    box_dist = 50/num_boxes
    total_shots = 0
    hoop_location = (0, 4)
    shot_matrix = np.empty([num_boxes,num_boxes])
    make_matrix = np.empty([num_boxes, num_boxes], dtype=float)
    for j in range(num_boxes):
        for i in range(num_boxes):
            total_shots_at_spot = len(df_player[df_player['loc_x'].between(-25+i*box_dist, -25+(i+1)*box_dist) &
                                 df_player['loc_y'].between(j*box_dist,(j+1)*box_dist)])
            current_block = df_player[df_player['loc_x'].between(-25 + i * box_dist, -25 + (i + 1) * box_dist) &
                                                df_player['loc_y'].between(j * box_dist, (j + 1) * box_dist)]
            total_makes_at_spot = len(current_block[current_block['shot_made'] == True])
            shot_matrix[j][i]= total_shots_at_spot + 1
            if(total_shots_at_spot == 0):
                make_matrix[j][i] = 0
            else:
                make_matrix[j][i] = (total_makes_at_spot / total_shots_at_spot)
            total_shots+=total_shots_at_spot
    shot_matrix = shot_matrix/total_shots
    text_matrix = shot_matrix
    if(scale == 'Sqrt'):
        shot_matrix = np.sqrt(shot_matrix)
    elif(scale == 'Log'):
        shot_matrix = np.log(shot_matrix)

    hover_text = np.array([["Field Goal %: {:.2%}<br>% of Total Shots: {:.2%}<br>Distance from hoop (ft): {:.2f}".format(
                            make_matrix[j][i], text_matrix[j][i], 
                            distance(-25 + i * box_dist + box_dist / 2, j * box_dist + box_dist / 2, hoop_location[0], hoop_location[1])) for i in range(num_boxes)] for j in range(num_boxes)])
    
    if(scale != 'Log' and scale != 'Sqrt'):
        scale_name = ''
    else:
        scale_name = scale

    heatmap = go.Heatmap(z=shot_matrix, text=hover_text, hoverinfo='text', colorscale = 'burgyl', colorbar={"title": scale_name + ' Shot Frequency', "orientation" :'h'})

    fig = go.Figure(data=[heatmap])
    
    court_image = PILImage.open("basketball_court.png")

    fig.update_layout(
    title={
        'text': player_name + " Shot Locations",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        images=[
        go.layout.Image(
            source=court_image,
            xref="x",
            yref="y",
            x=-10.64,
            y=25.65,
            sizex=46,
            sizey=29,
            sizing="stretch",
            opacity=.1,
            layer="above"
        )
    ],
    autosize=False,
    width=600,
    height=600)
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

database_type = os.getenv('db_type')  # e.g., 'mysql', 'sqlite'
db_username = os.getenv('db_username')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_port = os.getenv('db_port')
db_name = os.getenv('db_name')


engine = create_engine(f'{database_type}://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}',pool_pre_ping=True)

name_query = "SELECT DISTINCT player_name FROM basketball_shots"

players = pd.read_sql(name_query, engine)

player_list = players["player_name"].tolist()

seasons = ["2003-04","2004-05","2005-06", "2006-07", "2007-08","2008-09","2009-10","2010-11","2011-12","2012-13","2013-14",
           "2014-15","2015-16","2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]

app = Dash(__name__)

app.layout = app.layout = dash.html.Div(className='row', children=[
    dash.html.H1("Basketball Shot Data"),
    dash.html.H1("Year Comparison", style={'text-align': 'center'}),
    dash.html.Div(children=[
        dcc.Dropdown(seasons, id='years_dropdown_1', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-left': '9%', 'margin-right': '-12%'}),
        dcc.Dropdown(['None', 'Log', 'Sqrt'], id='scale_dropdown_1', style={'display': 'inline-block', 'width': '35%',
                                                                          'margin-right': '-14%'}),
        dcc.Dropdown(seasons, id='years_dropdown_2', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-right': '-12%'}),
        dcc.Graph(id='year_graph_1', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_season('2003-04', 'Log',engine)),
        dcc.Graph(id='year_graph_2', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_season('2023-24', 'Log', engine))
                  ]),

    dash.html.Div(children=[
        dcc.Graph(id='kde_year_graph_1', style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.2em'}, figure=distance_kde('2003-04', engine)),
        dcc.Graph(id='kde_year_graph_2', style={'display': 'inline-block', 'width': '45%'}, figure=distance_kde('2023-24', engine))
    ]),
    dash.html.H1("Player Comparison", style={'text-align': 'center'}),
    dash.html.Div(children=[
        dcc.Dropdown(player_list, id='players_dropdown_1', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-left': '9%', 'margin-right': '-12%'}),
        dcc.Dropdown(['None', 'Log', 'Sqrt'], id='scale_dropdown_2', style={'display': 'inline-block', 'width': '35%',
                                                                          'margin-right': '-14%'}),
        dcc.Dropdown(player_list, id='players_dropdown_2', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-right': '-12%'}),
        dcc.Graph(id='player_graph_1', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_player('Kyle Korver','Log', engine)),
        dcc.Graph(id='player_graph_2', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_player('LeBron James','Log', engine))
    ])
])

@callback(
    Output('year_graph_1', 'figure'),
    [Input('years_dropdown_1', 'value'),
    Input('scale_dropdown_1', 'value')]
)
def update_year_graph1(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_season('2003-04', 'Log', engine)
    elif value1 is None:
        return heatmap_by_season('2003-04', value2, engine)
    elif value2 is None:
        return heatmap_by_season(value1, 'Log', engine)
    else:
        return heatmap_by_season(value1, value2, engine)

@callback(
    Output('year_graph_2', 'figure'),
    [Input('years_dropdown_2', 'value'),
    Input('scale_dropdown_1', 'value')]

)
def update_year_graph2(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_season('2023-24', 'Log', engine)
    elif value1 is None:
        return heatmap_by_season('2023-24', value2, engine)
    elif value2 is None:
        return heatmap_by_season(value1, 'Log', engine)
    else:
        return heatmap_by_season(value1, value2, engine)

@callback(
    Output('kde_year_graph_1', 'figure'),
    Input('years_dropdown_1', 'value')
)
def update_kde1(value):
    if value is None:
        return distance_kde('2003-04', engine)
    else:
        return distance_kde(value, engine)

@callback(
    Output('kde_year_graph_2', 'figure'),
    Input('years_dropdown_2', 'value')
)
def update_kde2(value):
    if value is None:
        return distance_kde('2023-24', engine)
    else:
        return distance_kde(value, engine)

@callback(
    Output('player_graph_1', 'figure'),
    [Input('players_dropdown_1', 'value'),
    Input('scale_dropdown_2', 'value')]
)
def update_player_graph1(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_player('Kyle Korver', 'Log', engine)
    elif value1 is None:
        return heatmap_by_player('Kyle Korver', value2, engine)
    elif value2 is None:
        return heatmap_by_player(value1, 'Log', engine)
    else:
        return heatmap_by_player(value1, value2, engine)

@callback(
    Output('player_graph_2', 'figure'),
    [Input('players_dropdown_2', 'value'),
    Input('scale_dropdown_2', 'value')]
)
def update_player_graph2(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_player('LeBron James', 'Log', engine)
    elif value1 is None:
        return heatmap_by_player('LeBron James', value2, engine)
    elif value2 is None:
        return heatmap_by_player(value1, 'Log', engine)
    else:
        return heatmap_by_player(value1, value2, engine)

port = int(os.getenv('port', 4000))
app.run(host="0.0.0.0", port=port, debug=False)
