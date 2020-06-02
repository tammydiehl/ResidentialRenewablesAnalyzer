import os
import datetime
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# app = dash.Dash(__name__)


# Parser to help with the import of the date column, hopefully it works for all files..
# mydateparser = lambda x: datetime.datetime.strptime("2020-%s %s:00:00" % (x.split()[0], int(x.split()[-1].split(":")[0])-1), "%Y-%m/%d %H:%M:%S")

### traverse load data directory to get state and city information from files
### Comment out for now, just ran it once to create the file once
# load_files_avail = pd.DataFrame()
# for root, dirs, files in os.walk("residential_load_data"):
#     path = root.split(os.sep)
#     for file in files:
#         info_dict = {}
#         if file[:3] == "USA":
#             split_name = file.split("_")
#             info_dict["State"] = split_name[1]
#             info_dict["Locale"] = " ".join(split_name[2].split(".")[:-1])
#             info_dict["Load Level"] = split_name[-1][:-4]
#             info_dict["File Dir"] = root + "/" + file
#             load_files_avail = pd.concat([load_files_avail, pd.DataFrame(info_dict, index=[0])], ignore_index=True)
# load_files_avail.to_csv("residential_load_data/available_loads_summary.csv")

# import the loads that are available
available_loads = pd.read_pickle("residential_load_pickles/available_loads_summary.pkl")#, usecols=[1, 2, 3, 4])

# pre-make the list of dictionaries for dropdown menus
state_dropdown_options = [{'label': state, 'value': state} for state in available_loads["State"].unique()]
default_state = "GA"
locale_dropdown_options = [{'label': city, 'value': city} for city in available_loads[available_loads["State"] ==
                                                                                      default_state]["Locale"].unique()]
load_type_dropdown_options = [{'label': load_type, 'value': load_type} for load_type in available_loads[available_loads["State"] ==
                                                                                                        default_state]["Load Level"].unique()]

# Build simple Dash Layout
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Dropdown(
        options=state_dropdown_options,
        value='GA',
        id='chooseState',
        placeholder="Select state first",
    ),
    dcc.Dropdown(
        options=locale_dropdown_options,
        value="Atlanta-Hartsfield-Jackson Intl AP",
        id='chooseCity',
        placeholder="Select a city",
    ),
    dcc.Dropdown(
        options=load_type_dropdown_options,
        value="BASE",
        id='chooseLoadLevel',
        placeholder="Select home type",
    ),
    dcc.Markdown(id='fileDirectory'),
    dcc.Graph(id='loadGraph'),  # Graph that displays all data at minute resolution and allows selection of events,
    html.Div(default_state, id='current_state', style={'display': 'none'}),
])


# Create graph with data
@app.callback([Output('loadGraph', 'figure'),
               Output('fileDirectory', 'children'),
               Output('chooseCity', 'options'),
               Output('chooseLoadLevel', 'options'),
               Output('current_state', 'children')],
              [Input('chooseState', 'value'),
               Input('chooseCity', 'value'),
               Input('chooseLoadLevel', 'value')],
              [State('current_state', 'children')])
def event_plots(st, locl, load, current_st):
    # re-make the list of dictionaries for dropdown menus
    filter_state = available_loads[available_loads["State"] == st]
    load_type_dropdown_options = [{'label': load_type, 'value': load_type} for load_type in
                                  filter_state["Load Level"].unique()]
    locale_dropdown_options = [{'label': city, 'value': city} for city in filter_state["Locale"].unique()]

    if st == current_st:
        data_dir = available_loads[(available_loads["State"] == st) & (available_loads["Locale"] == locl) &
                                   (available_loads["Load Level"] == load)]["File Dir"].iloc[0]
    else:
        data_dir = available_loads[(available_loads["State"] == st) &
                                   (available_loads["Load Level"] == load)]["File Dir"].iloc[0]

    load_data = pd.read_pickle(data_dir)   #, date_parser=mydateparser, parse_dates=[0]).set_index('Date/Time')

    ### Tammy plot will replace this figure
    # Create a filled area plot using plotly graph_objects
    fig = go.Figure()
    # for cols in load_data:
    #     if ~(load_data[cols] == 0).all():
    fig.add_trace(
        go.Scatter(
            x=load_data.index,
            y=load_data,    # [cols]
            name="Total Load (kW)",
            mode='none',
            stackgroup='one'
        )
    )

    return fig, data_dir, locale_dropdown_options, load_type_dropdown_options, st


if __name__ == '__main__':
    app.run_server(debug=True)
