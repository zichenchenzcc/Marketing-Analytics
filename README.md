# Marketing Analytics

- First item
- Second item
- Third item
    - Indented item
    - Indented item
- Fourth item

> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.
>> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

# Heading level 1
## Heading level 2
###### Heading level 6
![Tux, the Linux mascot](/assets/images/tux.png)
1.  Open the file.
2.  Find the following code block on line 21:

        import dash 
        from dash.dependencies import Input, Output, State, MATCH
        import dash_table
        import dash_core_components as dcc
        import dash_html_components as html
        import plotly.express as px
        import pandas as pd
        import numpy as np
        import os

3.  Update the title to match the name of your website.


```python
python
import dash 
from dash.dependencies import Input, Output, State, MATCH
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import os

os.chdir("C:\\Users\\czc\\Desktop\\Python2\\Project 3\\happiness score")
df = pd.read_csv("covid_19_data_sub.csv")
df.iloc[:,0:8] = df.iloc[:,0:8].fillna(0)
df.iloc[:,8:18] = df.iloc[:,8:18].fillna(0)
column_list = [0,1,2,3,5,7,8,13,16]
df_group = df.iloc[:,column_list]
df_agg = df_group.groupby(['country']).agg({'new_cases':'sum','new_deaths':'sum'}).reset_index()

column_list1 = [0,1,2,8,13,16]
df_new = df.iloc[:,column_list1]
df_new = df_new.drop_duplicates().reset_index()
df_new = df_new.drop('index',axis=1)
df_new['Total Case'] = df_agg['new_cases']
df_new['Total Death'] = df_agg['new_deaths']
df_new.loc[df_new['Total Case'] == df_new['Total Case'].max()]
df_new = df_new.drop([188])
df_new['Logarithmic Total Case'] = np.log(df_new['Total Case']+1)
df_new.columns = ['ISO Code','Continent','Country','Population','GDP/Capita','Life Expectancy','Total Case','Total Death','Logarithmic Total Case']
df_new['GDP/Capita'] = df_new['GDP/Capita'].round(4)
columns = ['ISO Code','Continent','Country','Population','GDP/Capita','Life Expectancy','Total Case','Total Death']
dff = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,13,16]]
dff.columns = ['ISO Code','Continent','Country','Date','Total Case','New Case','Total Death','New Death','Population','Population Density','GDP/Capita','Life Expectancy']
dff = dff.drop(dff.loc[dff['Date']=='1/22/2020'].index)

app = dash.Dash(__name__) 
app.layout = html.Div([
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "selectable": True}
            if i == "Country" or i == "Total Case" or i == "Total Death"
            else {"name": i, "id": i, "selectable": True, "hideable": True}
            for i in columns
        ],
        data=df_new.to_dict('records'),  # the contents of the table
        editable=True,              # allow editing of data inside all cells
        filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
        row_selectable="multi",     # allow users to select 'multi' or 'single' rows
        row_deletable=True,         # choose if user can delete a row (True) or not (False)
        selected_columns=[],        # ids of columns that user selects
        selected_rows=[],           # indices of rows that user selects
        page_action="native",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=6,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': 95, 'maxWidth': 95, 'width': 95
        },
        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['ISO Code','Continent','Country']
        ],
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto'
        }
    ),
    html.Br(),
    html.Br(),
    html.Div(id='choromap-container'),
    html.Br(),
    html.Div([
    html.Div(children=[html.Button('Add Customized Chart', id='add-chart', n_clicks=0)]),
    html.Div(id='container', children=[])
            ])
])
```
