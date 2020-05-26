import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import datetime

mydateparser = lambda x: datetime.datetime.strptime("2020-%s %s:00:00" % (x.split()[0], int(x.split()[-1].split(":")[0])-1), "%Y-%m/%d %H:%M:%S")

df = pd.read_csv('residential_load_data/HIGH/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3_HIGH.csv',date_parser=mydateparser, parse_dates=[0]).set_index('Date/Time')



#Create a filled area plot using plotly graph_objects
fig = go.Figure()
for cols in df:
    if ~(df[cols] == 0).all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[cols],
                name=cols,
                mode='none',
                stackgroup='one'
            )
        )
plot(fig, filename="plots/load_plotter.html")

##Load app into heroku?
##Have all load together instead of breaking out the load would make it much faster