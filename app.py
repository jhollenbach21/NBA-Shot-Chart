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

def heatmap_by_season(df, season, scale):
    df_season = df[df['season'] == season]
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

def distance_kde(df, season):
    df_season = df[df['season']== season]
    hoop_location = (0, 4)
    shot_distances = pd.Series([
        distance(df_season['loc_x'][i], df_season['loc_y'][i], hoop_location[0], hoop_location[1]) 
        for i in range(len(df))
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

def heatmap_by_player(df, player_name, scale):
    df_player = df[df['player_name'] == player_name]
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

query = "SELECT * FROM basketball_shots"
df = pd.read_sql(query, engine)

app = Dash(__name__)

app.layout = app.layout = dash.html.Div(className='row', children=[
    dash.html.H1("Basketball Shot Data"),
    dash.html.H1("Year Comparison", style={'text-align': 'center'}),
    dash.html.Div(children=[
        dcc.Dropdown(df['season'].unique(), id='years_dropdown_1', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-left': '9%', 'margin-right': '-12%'}),
        dcc.Dropdown(['None', 'Log', 'Sqrt'], id='scale_dropdown_1', style={'display': 'inline-block', 'width': '35%',
                                                                          'margin-right': '-14%'}),
        dcc.Dropdown(df['season'].unique(), id='years_dropdown_2', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-right': '-12%'}),
        dcc.Graph(id='year_graph_1', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_season(df, '2003-04', 'Log')),
        dcc.Graph(id='year_graph_2', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_season(df, '2023-24', 'Log'))
    ]),
    dash.html.Div(children=[
        dcc.Graph(id='kde_year_graph_1', style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.2em'}, figure=distance_kde(df, '2003-04')),
        dcc.Graph(id='kde_year_graph_2', style={'display': 'inline-block', 'width': '45%'}, figure=distance_kde(df, '2023-24'))
    ]),
    dash.html.H1("Player Comparison", style={'text-align': 'center'}),
    dash.html.Div(children=[
        dcc.Dropdown(df['player_name'].unique(), id='players_dropdown_1', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-left': '9%', 'margin-right': '-12%'}),
        dcc.Dropdown(['None', 'Log', 'Sqrt'], id='scale_dropdown_2', style={'display': 'inline-block', 'width': '35%',
                                                                          'margin-right': '-14%'}),
        dcc.Dropdown(df['player_name'].unique(), id='players_dropdown_2', style={'display': 'inline-block', 'width': '45%',
                                                                        'margin-right': '-12%'}),
        dcc.Graph(id='player_graph_1', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_player(df,'Kyle Korver','Log')),
        dcc.Graph(id='player_graph_2', style={'display': 'inline-block', 'width': '48%'}, figure=heatmap_by_player(df,'LeBron James','Log'))
    ])
])

@callback(
    Output('year_graph_1', 'figure'),
    [Input('years_dropdown_1', 'value'),
    Input('scale_dropdown_1', 'value')]
)
def update_year_graph1(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_season(df, '2003-04', 'Log')
    elif value1 is None:
        return heatmap_by_season(df, '2003-04', value2)
    elif value2 is None:
        return heatmap_by_season(df, value1, 'Log')
    else:
        return heatmap_by_season(df, value1, value2)

@callback(
    Output('year_graph_2', 'figure'),
    [Input('years_dropdown_2', 'value'),
    Input('scale_dropdown_1', 'value')]

)
def update_year_graph2(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_season(df, '2023-24', 'Log')
    elif value1 is None:
        return heatmap_by_season(df, '2023-24', value2)
    elif value2 is None:
        return heatmap_by_season(df, value1, 'Log')
    else:
        return heatmap_by_season(df, value1, value2)

@callback(
    Output('kde_year_graph_1', 'figure'),
    Input('years_dropdown_1', 'value')
)
def update_kde1(value):
    if value is None:
        return distance_kde(df, '2003-04')
    else:
        return distance_kde(df, value)

@callback(
    Output('kde_year_graph_2', 'figure'),
    Input('years_dropdown_2', 'value')
)
def update_kde2(value):
    if value is None:
        return distance_kde(df, '2023-24')
    else:
        return distance_kde(df, value)

@callback(
    Output('player_graph_1', 'figure'),
    [Input('players_dropdown_1', 'value'),
    Input('scale_dropdown_2', 'value')]
)
def update_player_graph1(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_player(df, 'Kyle Korver', 'Log')
    elif value1 is None:
        return heatmap_by_player(df, 'Kyle Korver', value2)
    elif value2 is None:
        return heatmap_by_player(df, value1, 'Log')
    else:
        return heatmap_by_player(df, value1, value2)

@callback(
    Output('player_graph_2', 'figure'),
    [Input('players_dropdown_2', 'value'),
    Input('scale_dropdown_2', 'value')]
)
def update_player_graph2(value1,value2):
    if value1 is None and value2 is None:
        return heatmap_by_player(df, 'LeBron James', 'Log')
    elif value1 is None:
        return heatmap_by_player(df, 'LeBron James', value2)
    elif value2 is None:
        return heatmap_by_player(df, value1, 'Log')
    else:
        return heatmap_by_player(df, value1, value2)

port = int(os.getenv('port')
app.run(host="0.0.0.0", port=port, debug=False)

