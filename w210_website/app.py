import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

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
    'showbackground': True,
    'backgroundcolor': '#171b26',
    }

  for i in range(len(genre_dict)):

      trace_set.append(go.Scatter3d(x=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['tsne-3d-one'], 
                                    y=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['tsne-3d-two'],
                                    z=full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['tsne-3d-three'],
                                    mode='markers', name=genre_dict[i],
                                    hovertemplate = '<i>Track ID</i>: %{text}'
                                                    '<br><i>x</i>: %{x}'
                                                    '<br><i>y</i>: %{y}'
                                                    '<br><i>z</i>: %{z}',
                                    marker=dict(size=3, 
                                                line=dict(width=0.1),
                                                color=genre_colors[i],
                                                opacity=1),
                                    text = full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['track_id'] 
                                    #      full_set_df[full_set_df['track_genre_top']==genre_dict[i]]['track_title']
                                   )
                      )

  layout = go.Layout(margin=dict(l=50, r=0, b=0, t=50),
                     title= 'T-SNE 3D Scatter Plot of Song Embeddings',
                     font= {'size': 12, 'color': 'white'},
                     plot_bgcolor= axis_template['backgroundcolor'],
                     paper_bgcolor= axis_template['backgroundcolor'],
                     hovermode= 'closest',
                     height=800,
                     scene = dict(
                                 aspectratio = {'x': 1, 'y': 1.2, 'z': 1},
                                 ),
                     legend = dict(itemsizing='constant',
                                  orientation='h',
                                  xanchor='auto',
                                  yanchor='auto'
                                  ),
                     clickmode= 'select',
                     uirevision= 'same',
                    )

  return trace_set, layout

# Running the functions to load the TSNE results and get the trace and layout for graphing
full_set_df = load_tsne_results(full_set_df, embedding_dir, results_file=results_file)
trace_set, layout = gen_tsne_3Dplot(full_set_df, genre_dict, embedding_dir, results_file)

# Convert the updated full_set_df into json
# This is needed to pass into the intermediate Div as children
jsonified_full_set_df = full_set_df.to_json(orient='split')


# Start build of Dash layout
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
  children=[

    html.Div(className='two-thirds column app__left__section', 
      children=[

      html.H2(
        children='Exploring Free Music Archive with Song Embeddings'
        ),

      dcc.Graph(id='TSNE',
        figure={
            'data': trace_set,
            'layout': layout,
          }
        ),
      ]),

    html.Div(className='one-third column app__right__section', 
      children=[

      html.H3(children='Let\'s Get Started!'),

      html.Div(children='First select a song from the graph on the left...'),

      html.Div(id='current-song',
        children=[
        html.H6(children='Current Song Selection: '),
        html.Table(id='current-song-table',
          children=[
          html.Tr([html.Th('Title'), html.Th('Artist')])
          ] + 
          [
          html.Tr([
            html.Td(id='current-song-title', children='-'), 
            html.Td(id='current-song-artist', children='-')])
          ])
        ]),

      html.H6(children='Select a filter for Euclidean distance:'),
      
      html.Div(style={'margin-bottom': 25},
        children=[
        dcc.Slider(id='distance-slider',
          min=0,
          max=50,
          value=50,
          marks={str(dist):str(dist) for dist in np.arange(0,51,5)},
          step=0.5,
          ),
        ]),

      html.Div(
      children=[
        html.Div(id='distance-slider-feedback', 
          style = {'display': 'inline-block', 'margin-left': 5, 'width': '65%'}),
        html.Button('Reset', id='reset-plot-button'),
        ]),      

      html.Div(id='nearest-songs',
        children=[
        html.H6(children='Nearest Songs to Current Selection: '),
        html.Label(id='nearest-songs-attr', children='-'),
        ]),

      ]),

    # Intermediate Hidden Divs for passing data
    html.Div(id='full-set-df-intermediate', children=jsonified_full_set_df, style={'display': 'none'}),
    html.Div(id='song-info-intermediate', children='-', style={'display': 'none'}),
    html.Div(id='sub-set-df-intermediate', children='-', style={'display': 'none'}),
    dcc.Graph(id='original-TSNE',
        figure={
            'data': trace_set,
            'layout': layout,
          },
        style={'display': 'none'}
        ),
    html.Div(id='calc-nn-in-progress', children=0, style={'display': 'none'}),

  ])


# Implementing callbacks 

# Callback of song data from the clicked data point on TSNE graph, and 
# Saving the song data in intermediate div from the clicked data point on TSNE graph

@app.callback(
  Output('song-info-intermediate', 'children'),
  [Input('TSNE', 'clickData')],
  [State('full-set-df-intermediate', 'children')]
  )
def display_song_on_click(clickData, jsonified_full_set_df):
  full_set_df = pd.read_json(jsonified_full_set_df, orient='split')

  if clickData != None:
    click_song_id = clickData['points'][0]['text']
    updated_song_info = {
      'track_id' : click_song_id,
      'song' : full_set_df[full_set_df['track_id']==click_song_id]['track_title'].item(),
      'artist' : full_set_df[full_set_df['track_id']==click_song_id]['artist_name'].item(),
      'genre' : full_set_df[full_set_df['track_id']==click_song_id]['track_genre_top'].item(),    
      }
    return json.dumps(updated_song_info)
  else:
    pass


# Display current selection in current song table
@app.callback(
  [Output('current-song-title', 'children'),
  Output('current-song-artist', 'children')],
  [Input('song-info-intermediate', 'children')]
  )
def show_current_song_in_table(current_song_info):
  try:
    song_info = json.loads(current_song_info)
    song_title = json.dumps(song_info['song'])
    artist_name = json.dumps(song_info['artist'])
    return song_title, artist_name
  except:
    return '-', '-'

      
# Callback of Euclidean distance from the distance slider
@app.callback(
    Output('distance-slider-feedback', 'children'),
    [Input('distance-slider', 'value')])
def update_distance_slider_feedback(value):
    return 'Applying filter distance of {}.'.format(value)


# Callback for updating a jsonified subsample of full_set_df 
# based on selected Euclidean distance and selected song
@app.callback(
  [Output('sub-set-df-intermediate', 'children'),
  Output('nearest-songs-attr', 'children')],  
  [Input('song-info-intermediate', 'children'),
  Input('distance-slider', 'value')],
  [State('full-set-df-intermediate', 'children')]
  )
def save_sub_set_df(current_song_info, distance, jsonified_full_set_df):
  try:
    song_info = json.loads(current_song_info)
    click_song_id = song_info['track_id']
    full_set_dist_temp = pd.read_json(jsonified_full_set_df, orient='split')
    
    song_tsne = full_set_dist_temp[full_set_dist_temp['track_id']==click_song_id][['tsne-3d-one', 'tsne-3d-two', 'tsne-3d-three']]
    full_set_tsne = full_set_dist_temp[['tsne-3d-one', 'tsne-3d-two', 'tsne-3d-three']]
    
    # Getting the distance between all songs and the selected song
    full_set_dist_temp['song_distance'] = [np.linalg.norm(full_set_tsne.iloc[i]-song_tsne)
     for i in range(full_set_tsne.shape[0])]
    
    # Getting a sorted subset of the songs which are within the distance in the slider
    sub_set_df = full_set_dist_temp[full_set_dist_temp['song_distance']
                                    <= distance].sort_values(by='song_distance')  
    # Sending back the song all nearest neighbors
    jsonified_sub_set_df = sub_set_df[:].to_json(orient='split')

  except:
    jsonified_sub_set_df = '-'

  return jsonified_sub_set_df, jsonified_sub_set_df


# Callback for updating the TSNE graph 
# based on the selected Euclidean distance and selected song
@app.callback(
  Output('TSNE', 'figure'),
  [Input('reset-plot-button', 'n_clicks'),
  Input('distance-slider', 'value'),
  Input('sub-set-df-intermediate', 'children')],
  [State('song-info-intermediate', 'children'),
  State('TSNE', 'figure'),
  State('original-TSNE', 'figure')]
  )
def update_TSNE(reset_n_clicks, distance, jsonified_sub_set_df,
current_song_info, figure, original_figure):
  trace_set = original_figure['data']
  layout = figure['layout']
  if (jsonified_sub_set_df != '-') and (current_song_info != '-'):
    nn_df = pd.read_json(jsonified_sub_set_df, orient='split')
    song_id = nn_df.iloc[0]['track_id']
    # Include a try and except for fault tolerance
    try:
      # Activate filtering when distance is not equal to the max value
      if (distance != 50):
        # Update key/value pairs in 'layout'
        layout['title'] = f'Nearest Neighbors to Selected Song with Distance {distance}'
        layout['scene']['xaxis'] = dict(range=[min(nn_df['tsne-3d-one']), max(nn_df['tsne-3d-one'])])
        layout['scene']['yaxis'] = dict(range=[min(nn_df['tsne-3d-two']), max(nn_df['tsne-3d-two'])])
        layout['scene']['zaxis'] = dict(range=[min(nn_df['tsne-3d-three']), max(nn_df['tsne-3d-three'])])
      elif (reset_n_clicks != None):
        # Reset called, reload the original layout
        layout = original_figure['layout']   
    except:
      pass
    # Add annotation to selected song, 
    # Note: putting this outside keeps annotation in both filtering and reset calls
    layout['scene']['annotations'] = [dict(
      showarrow=True,
      x=nn_df.iloc[0]['tsne-3d-one'],
      y=nn_df.iloc[0]['tsne-3d-two'],
      z=nn_df.iloc[0]['tsne-3d-three'],
      text='Selected Song: {}'.format(nn_df.iloc[0]['track_title']),
      opacity=0.7,
      arrowcolor='white',
      )]
  return {'data': trace_set, 'layout': layout}


# Callback to reset the filters by reseting the distance slider to max value
@app.callback(
  Output('distance-slider', 'value'),
  [Input('reset-plot-button', 'n_clicks')]
  )
def reset_filter(reset_n_clicks):
  return 50



if __name__ == '__main__':
    app.run_server(debug=True)