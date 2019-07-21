import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
import plotly.express as px

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

# Reading in the tracks.csv file
tracks_df = pd.read_csv("./data/tracks.csv", sep=',', index_col=0, header=[0,1])

train_set_df = tracks_df[tracks_df[('set','split')]=='training']
val_set_df = tracks_df[tracks_df[('set','split')]=='validation']
test_set_df = tracks_df[tracks_df[('set','split')]=='test']

# Generating the concatenated (train, val, test) set of track ids and properties 
full_set_df = pd.concat([train_set_df, val_set_df, test_set_df])

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

      trace_set.append(go.Scatter3d(x=full_set_df[full_set_df[('track','genre_top')]==genre_dict[i]]["tsne-3d-one"], 
                                    y=full_set_df[full_set_df[('track','genre_top')]==genre_dict[i]]["tsne-3d-two"],
                                    z=full_set_df[full_set_df[('track','genre_top')]==genre_dict[i]]["tsne-3d-three"],
                                    mode='markers', name=genre_dict[i],
                                    hovertemplate = "<i>Title</i>: %{text}"
                                                    "<br><i>x</i>: %{x}"
                                                    "<br><i>y</i>: %{y}"
                                                    "<br><i>z</i>: %{z}",
                                    marker=dict(size=3, 
                                                line=dict(width=0.1),
                                                color=genre_colors[i],
                                                opacity=1),
                                    text = full_set_df[full_set_df[('track','genre_top')]==genre_dict[i]][('track','title')]
                                   )
                      )

  layout = go.Layout(margin=dict(l=100, r=0, b=0, t=50),
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
                    )
 
  return trace_set, layout

full_set_df = load_tsne_results(full_set_df, embedding_dir, results_file=results_file)

trace_set, layout = gen_tsne_3Dplot(full_set_df, genre_dict, embedding_dir, results_file)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
  children=[

    html.Div(className="two-thirds column app__left__section", 
      children=[

      html.H1(
        children='Hello Dash'
        ),

      html.Div(
        children= "Dash: A web application framework for Python."
        ),

      dcc.Graph(id='TSNE',
          figure={
              'data': trace_set,
              'layout': layout
          }
          ),      
      ]),

    html.Div(className="one-third column app__right__section", 
      children=[

      html.Div(children="Right side"),
      
      html.Label(
        children='What song genres are you interested in?'
        ),

      dcc.Dropdown(
        options=[
        {'label': genre_dict[i], 'value': i} for i in range(len(genre_dict))
        ],
        multi=True
        ),
      
      html.Label(
        children='Euclidean distance slider'
        ),
      
      dcc.Slider(id='distance-slider',
          min=0,
          max=50,
          value=50,
          marks={str(dist):str(dist) for dist in np.arange(0,51,5)},
          step=1
          ),
      ])

  ])




if __name__ == '__main__':
    app.run_server(debug=True)