import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import json

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Set embedding directory
embedding_dir = './data'
results_file = 'tsne_results_set1_p50'

genre_dict = {0 : 'Hip-Hop',
              1 : 'Pop',
              2 : 'Folk',
              3 : 'Experimental',
              4 : 'Rock',
              5 : 'International',
              6 : 'Electronic',
              7 : 'Instrumental'}

# Reading in the full_set_df_collapse_header.csv file
full_set_df = pd.read_csv(f'{embedding_dir}/full_set_df_collapse_header.csv', sep=',', index_col=0)

def load_tsne_results(full_set_df, embedding_dir, results_file):
  '''
  Loads TSNE results
  
  Inputs
  ------
  full_set_df: pandas DF with column names:
      - ('track','genre_top')
      - ('track','title')
  genre_dict: dictionary of genres in the format, e.g.: {0: 'Hip-hop', 1: 'Pop', ...}    
  embedding_dir: directory containing the embeddings and results file
  results_file: the tsne results filename
  
  Returns
  -------
  full_set_df with additional columns:
      - 'tsen-3d-one'
      - 'tsen-3d-two'
      - 'tsen-3d-three'
  '''
  
  # Load results
  tsne_results = np.load(f'{embedding_dir}/{results_file}.npy')
  
  # Set the dataframes to include the tsne_results
  full_set_df['tsne-3d-one'] = tsne_results[:,0]
  full_set_df['tsne-3d-two'] = tsne_results[:,1]
  full_set_df['tsne-3d-three'] = tsne_results[:,2]
  
  return full_set_df

def gen_tsne_3Dplot(full_set_df, genre_dict, embedding_dir, results_file):
  '''
  Generates the trace set and layout files for 3D plotly scatter plot
  
  Inputs
  ------
  full_set_df: pandas DF with column names:
      - ('track','genre_top')
      - ('track','title')
      - 'tsen-3d-one'
      - 'tsen-3d-two'
      - 'tsen-3d-three'
  genre_dict: dictionary of genres in the format, e.g.: {0: 'Hip-hop', 1: 'Pop', ...}    
  embedding_dir: directory containing the embeddings and results file
  results_file: the tsne results filename
  
  Returns
  -------
  trace_set, layout
  '''
      
  trace_set = []

  genre_colors = px.colors.qualitative.Plotly

  axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
    }

  for i in range(len(genre_dict)):

      trace_set.append(go.Scatter3d(x=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]["tsne-3d-one"], 
                                    y=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]["tsne-3d-two"],
                                    z=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]["tsne-3d-three"],
                                    mode='markers', name=genre_dict[i],
                                    hovertemplate = "<i>Title</i>: %{text}"
                                                    "<br><i>x</i>: %{x}"
                                                    "<br><i>y</i>: %{y}"
                                                    "<br><i>z</i>: %{z}",
                                    marker=dict(size=3, 
                                                line=dict(width=0.1),
                                                color=genre_colors[i],
                                                opacity=1),
                                    text = full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['track_title']
                                   )
                      )

  layout = go.Layout(margin=dict(l=50, r=0, b=0, t=50),
                     title= f'T-SNE 3D Plot - n_components=3, results={results_file}',
                     font= {"size": 12, "color": "white"},
                     plot_bgcolor= axis_template["backgroundcolor"],
                     paper_bgcolor= axis_template["backgroundcolor"],
                     hovermode= 'closest',
                     height=800,
                     scene = dict(xaxis = axis_template,
                                 yaxis = axis_template,
                                 zaxis = axis_template,
                                 aspectratio = {"x": 1, "y": 1.2, "z": 1},
                                 ),
                     legend = dict(itemsizing="constant",
                                  orientation="h",
                                  xanchor="auto",
                                  yanchor="auto"
                                  ),
                     clickmode= 'select',
                    )
 
  return trace_set, layout

# Running the functions to load the TSNE results and get the trace and layout for graphing
full_set_df = load_tsne_results(full_set_df, embedding_dir, results_file=results_file)
trace_set, layout = gen_tsne_3Dplot(full_set_df, genre_dict, embedding_dir, results_file)

# Convert the updated full_set_df inot json
# This is needed to pass into the intermediate Div as children
jsonified_full_set_df = full_set_df.to_json(orient='split')


# Start build of Dash layout
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
  children=[

    html.Div(className="two-thirds column app__left__section", 
      children=[

      html.H1(
        children='Hello Dash'
        ),

      html.Div(children= "Dash: A web application framework for Python."),

      dcc.Graph(id='TSNE',
        figure={
            'data': trace_set,
            'layout': layout,
          }
        ),

      ]),

    html.Div(className="one-third column app__right__section", 
      children=[

      html.Div(children="Right side"),
      
      html.Label(children='What song genres are you interested in?'),

      dcc.Dropdown(
        options=[
        {'label': genre_dict[i], 'value': i} for i in range(len(genre_dict))
        ],
        multi=True
        ),
      
      html.Label(children='Euclidean distance slider'),
      
      html.Div(style={'margin-bottom': 40},
        children=[
        dcc.Slider(id='distance-slider',
          min=0,
          max=50,
          value=50,
          marks={str(dist):str(dist) for dist in np.arange(0,51,5)},
          step=1
          ),
        ]),

      html.Div(id='distance-slider-feedback'),

      html.Div(id='current-song',
        children=[
        html.Label(children='Current Song Selection: '),
        html.Label(id='current-song-attr'),
        dash_table.DataTable(id='current-song-info-table'),      
        ]),

      ]),

    html.Div(id='full-set-df-intermediate', children=jsonified_full_set_df, style={'display': 'none'}),
    html.Div(id='song-info-intermediate', style={'display': 'none'}),

  ])


# Implementing callbacks 

# Callback of song data from the clicked data point on TSNE graph, and 
# Saving the song data in intermediate div from the clicked data point on TSNE graph

@app.callback(
  Output('song-info-intermediate', 'children'),
  [Input('TSNE', 'clickData'),
  Input('full-set-df-intermediate', 'children')]
  )
def display_song_on_click(clickData, jsonified_full_set_df):
  full_set_df = pd.read_json(jsonified_full_set_df, orient='split')
  
  if clickData != None:
    click_song = clickData['points'][0]['text']
    updated_song_info = {
      'track_id' : full_set_df[full_set_df['track_title']==click_song]['track_id'].item(),
      'song' : click_song,
      'artist' : full_set_df[full_set_df['track_title']==click_song]['artist_name'].item(),
      'genre' : full_set_df[full_set_df['track_title']==click_song]['track_genre_top'].item(),
      }
  else:
    updated_song_info = {}

  return json.dumps(updated_song_info)

# Test print of clicked song info
@app.callback(
  Output('current-song-attr', 'children'),
  [Input('song-info-intermediate', 'children')]
  )
def save_song_info(updated_song_info):
  return updated_song_info

      

# Callback of Euclidean distance from the distance slider
@app.callback(
    Output('distance-slider-feedback', 'children'),
    [Input('distance-slider', 'value')])
def update_distance_slider_feedback(value):
    return 'You have selected a Euclidean distance of "{}"'.format(value)

# Update functions

# def generate_table(dataframe, song_id=2):
#     return html.Table(
#         # Header
#         [html.Tr([html.Th(col) for col in dataframe.columns])] +

#         # Body
#         [html.Tr([
#             html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#         ]) for i in range(min(len(dataframe), max_rows))]
#     )


if __name__ == '__main__':
    app.run_server(debug=True)