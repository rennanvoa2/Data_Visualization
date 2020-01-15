import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.colors
import random
from copy import deepcopy

#if this link is broken go to https://github.com/rennanvoa2/Data_Visualization and get movies.csv raw link
#or use the movies.csv that is inside of the data folder
df = pd.read_csv('https://raw.githubusercontent.com/rennanvoa2/Data_Visualization/master/movies.csv', sep=';')
drop_list = ['company','released','rating', 'votes','writer']
df.drop(drop_list, inplace=True, axis=1)

df['profit'] = df['gross'] - df['budget']

df.rename(columns={'gross':'revenue'},inplace=True)
genre_options = [dict(label=genre, value=genre) for genre in df['genre'].unique()]
country_options2 = [dict(label=country, value=country) for country in df['country'].unique()]
country_options2.insert(0,{'label':"All", 'value':'All'})
genre_options.insert(0,{'label':"All", 'value':'All'})

#df = df.loc[(df['country'] == country) & (df['year'] == year) & (df['genre'] == genre)]
app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.H1('30 Years of Movie Industry')
        ]),

    html.Div([
        html.Div([
            html.Label('Year'),
            dcc.RangeSlider(id='year_slider',value=[1986,2016], min=df['year'].min(), max=df['year'].max(), marks={str(i): '{}'.format(str(i)) for i in [1986,1990, 1995, 2000, 2005, 2010, 2016]},step=1)],className='column1 filters'),

        html.Div([
            html.Label('Country'),
            dcc.Dropdown(id='country_drop',options=country_options2,value=['All'],multi=True)], className='column1 filters'),

        html.Div([
            html.Label('Genre'),
            dcc.Dropdown(id='genre_drop',options=genre_options,value=['All'],multi=True)], className='column1 filters'),], className='row'),

    html.Div([dcc.Graph(id='mov_inv')], className='pretty'),

    html.Div([

        html.Div([dcc.Graph(id='scatter_chart')],className='column3 pretty'),

        html.Div([dcc.Graph(id='top_10')], className='column3 pretty'),

    ], className='row'),

    html.Div([

        html.Div([dcc.Graph(id='wordcloud')],className='column3 pretty'),

        html.Div([dcc.Graph(id='direc')], className='column3 pretty'),

    ], className='row'),

    html.Div([dcc.Graph(id='genre_years')], className='pretty')

])





@app.callback(
    [
        Output("mov_inv", 'figure'),
        Output("top_10", 'figure'),
        Output("scatter_chart", 'figure'),
        Output("wordcloud", 'figure'),
        Output("genre_years", 'figure'),
        Output("direc", 'figure')
    ]
    ,
    [
        Input("year_slider", "value"),
        Input("country_drop", "value"),
        Input("genre_drop", "value"),
    ]
)

def update_fig(year_slider, country_drop, genre_drop ):

#_____________________________________________________________RUNTIME X PROFIT __________________________________________________________________

    df_loc = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]



    if  'All' in country_drop and 'All' in genre_drop:
        df_loc = df
    elif 'All' in country_drop:
        df_loc = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_loc = df.loc[df['country'].isin(country_drop)]

    df_loc = df_loc[df_loc['year'].isin(range(year_slider[0],year_slider[-1]))]
    a = df_loc.profit.values/1000000
    fig_scatter = go.Figure(data=[go.Scatter(
        x=df_loc.runtime.values,
        y=df_loc.profit.values,

        mode='markers',
        text= df_loc.name.values,
        marker=dict(
            color=df_loc.score.values,
            size=(df_loc.score)/7,
            showscale=True,
            colorbar=dict(title="Average Rating<br> &nbsp;",
                titleside="top",
                tickmode="array",
                tickvals=[0,15,30,50,70,100],
                ticks="outside"),
            ),
                
        hovertemplate = 'Minutes: %{x}<br> Profit: %{y:$.2f}<br> Movie Name: %{text}',

    )])


    fig_scatter.update_layout(
        title='Run Time and Profit',
        xaxis=dict(title='Run Time (Minute)',
                titlefont_size=16,
                tickfont_size=14,
        ),
        yaxis=dict(title='Profit (US Dollar)',
                titlefont_size=16,
                tickfont_size=14,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )



#_____________________________________________________________Invest and Return __________________________________________________________________


    df_mov_invest = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]



    if  'All' in country_drop and 'All' in genre_drop:
        df_mov_invest = df
    elif 'All' in country_drop:
        df_mov_invest = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_mov_invest = df.loc[df['country'].isin(country_drop)]

        df_mov_invest = df_mov_invest[df_mov_invest['year'].isin(range(year_slider[0],year_slider[-1]))]

    df_line = df_mov_invest[['country', 'year','genre', 'budget', 'revenue', 'profit']]
    df_line = df_line[(df_line['year'] >= year_slider[0]) & (df_line['year'] <= year_slider[-1])]

    df_line = df_line.groupby('year').sum().reset_index()
    budget = df_line['budget']
    revenue = df_line['revenue']
    profit = df_line['profit']

    fig_mov_invest = go.Figure()
    fig_mov_invest.add_trace(go.Scatter(x=df_line['year'],
                            y=budget,
                            name='Budget',
                            line=dict(color='rgb(250,224,30)', width=5)
                            )
                )
    fig_mov_invest.add_trace(go.Scatter(x=df_line['year'],
                            y=revenue,
                            name='Revenue',
                            line=dict(color='rgb(232,10,147)', width=5)
                            )
                )
    fig_mov_invest.add_trace(go.Scatter(x=df_line['year'],
                            y=profit,
                            name='Profit',
                            line=dict(color="rgb(78,13,157)", width=5)
                            )
                )

    fig_mov_invest.update_layout(
        title='Movie Investment and Return',
        xaxis=dict(title='Year',
                titlefont_size=16,
                tickfont_size=12,
        ),
        yaxis=dict(title='(in US Dollar)',
                titlefont_size=16,
                tickfont_size=12,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


#_____________________________________________________________TOP10 Mov x Profit __________________________________________________________________

    df_top10 = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]



    if  'All' in country_drop and 'All' in genre_drop:
        df_top10 = df
    elif 'All' in country_drop:
        df_top10 = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_top10 = df.loc[df['country'].isin(country_drop)]

        df_top10 = df_top10[df_top10['year'].isin(range(year_slider[0],year_slider[-1]))]

    df_bar = df_top10
    df_bar = df_bar[(df_bar['year'] >= year_slider[0]) & (df_bar['year'] <= year_slider[1])]

    df_bar = df_bar.nlargest(10,'score')  # Top 10 movies by score

    x_bar = df_bar['name']
    y_bar = df_bar['profit']

    fig_top10 = go.Figure()
    fig_top10 = px.bar(df_bar, x=x_bar, y=y_bar,
                hover_data=['score', 'profit'],
                color='score',
                labels={'profit':'Profit', 'name': 'Movie'}, height=600)

    fig_top10.update_layout(
        title='Top Movies (by Score) and Profit',
        xaxis=dict(title='Movie',
                titlefont_size=16,
                tickfont_size=12,
        ),
        yaxis=dict(title='Profit (US Dollar)',
                titlefont_size=16,
                tickfont_size=12,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig_top10.update_xaxes(automargin=True)

#_____________________________________________________________WORDCLOUD __________________________________________________________________

    df_word = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]



    if  'All' in country_drop and 'All' in genre_drop:
        df_word = df
    elif 'All' in country_drop:
        df_word = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_word = df.loc[df['country'].isin(country_drop)]

        df_word = df_word[df_word['year'].isin(range(year_slider[0],year_slider[-1]))]

    df_cloud = df_word
    df_cloud = df_cloud[(df_cloud['year'] >= year_slider[0]) & (df_cloud['year'] <= year_slider[-1])]
    star_rev = df_cloud[['star', 'revenue']].groupby('star').sum().reset_index().nlargest(15,'revenue')

    words = star_rev.star.values

    revenue = star_rev.revenue.values

    lower, upper = 15, 45
    frequency = [((x - min(revenue)) / (max(revenue) - min(revenue))) * (upper - lower) + lower for x in revenue]

    percent = star_rev.revenue.values / star_rev.revenue.values.sum()

    lenth = len(words)
    col = ['#fcec03','#f5a122','#f58664','#8603ad','#ad038b','#011a73','#271f73','#881c91','#b80f88','#bf5426']
    colors = [col[random.randrange(0, 9)] for i in range(lenth)]

    data = go.Scatter(
    x=list(range(lenth)),
    y=random.choices(range(lenth), k=lenth),
    mode='text',
    text=words,
    hovertext=['{0}\n\n Revenue: {1}\n\n Percentage: {2}'.format(w, r, format(p, '.2%')) for w, r, p in zip(words, revenue, percent)],
    hoverinfo='text',
    textfont={'size': frequency, 'color': colors})
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

    fig_wordcloud = go.Figure(data=[data], layout=layout)

    fig_wordcloud.update_layout(
        title='Stars with the highest box office totals',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )



#_____________________________________________________________Genre x Years __________________________________________________________________

    df_genre_years = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]



    if  'All' in country_drop and 'All' in genre_drop:
        df_genre_years = df
    elif 'All' in country_drop:
        df_genre_years = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_genre_years = df.loc[df['country'].isin(country_drop)]

    df_genre_years = df_genre_years[(df_genre_years['year'] >= year_slider[0]) & (df_genre_years['year'] <= year_slider[-1])]

    df_gy = df_genre_years

    fig_gy = go.Figure()

    if 'All' in genre_drop:
        df_gy_all = deepcopy(df_gy.groupby('year').count().reset_index())

        fig_gy.add_trace(go.Scatter(x=df_gy_all['year'],
                                y=df_gy_all['genre'],
                                line=dict(color='rgb(250,224,30)', width=5),
                                name='All Genres',
                                hovertemplate = 'Year: %{x}<br> Quantity: %{y}',

                                )
                    )

    else:
        for i,g in enumerate(genre_drop):
            df_gen = df_genre_years.loc[(df_genre_years['genre'] == g)]
            df_gen = df_gen.groupby('year').count().reset_index()

            fig_gy.add_trace(go.Scatter(x=df_gen['year'],
                                    y=df_gen['genre'],
                                    line=dict(width=5),
                                    name=g,
                                    hovertemplate = 'Year: %{x}<br> Quantity: %{y}',

                                    )
                        )


    fig_gy.update_layout(
        title='Production of Movies by Genre Over Years',
        xaxis=dict(title='Year',
                titlefont_size=16,
                tickfont_size=12,
        ),
        yaxis=dict(title='Number of Movies by Genre',
                titlefont_size=16,
                tickfont_size=12,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


#_____________________________________________________________Director x Profit __________________________________________________________________

    df_director = df.loc[df['country'].isin(country_drop) & df['genre'].isin(genre_drop)]

    if  'All' in country_drop and 'All' in genre_drop:
        df_director = df
    elif 'All' in country_drop:
        df_director = df.loc[df['genre'].isin(genre_drop)]
    elif 'All' in genre_drop:
        df_director = df.loc[df['country'].isin(country_drop)]

    df_director = df_director[(df_director['year'] >= year_slider[0]) & (df_director['year'] <= year_slider[-1])]

    name_counts = df_director['director'].value_counts().to_dict()
    df_director['count'] = df_director['director'].map(name_counts)
    df_director = df_director.groupby('director').mean().reset_index()
    df_director['count'] = df_director['count'].astype(int)
    df_director['dc'] = df_director['director'].map(str) + " (" + df_director['count'].map(str) + ")"
    top15director = df_director.sort_values(ascending = False, by = 'profit')[['dc','profit']].head(15)

    data_direc = [go.Bar(
                x=top15director['profit'],
                y=top15director['dc'],
                orientation = 'h',
                marker=dict(
                color='rgb(145, 16, 113)'
            )
    )]

    layout_direc = dict(
            title='Average Profit by Directors',
            margin=go.Margin(
            l=210,
            r=100,
            pad=1),
            xaxis=dict(
                title='Average Profit'
            ),

            yaxis=dict(
                title='Director (Number of Movies)',
                tickfont=dict(
                    size=12,
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )


    fig_direc = go.Figure(data = data_direc, layout = layout_direc)




    return fig_mov_invest, \
        fig_scatter, \
            fig_top10, \
                fig_wordcloud, \
                    fig_gy, \
                        fig_direc


if __name__ == '__main__':
    app.run_server(debug=True, port=4051)