import pandas as pd
import requests
import json

import plotly.graph_objs as go
from plotly.offline import plot



def PlotData(df, buy_signals=False, sell_signals=False, plot_title: str = "",
             trends=False,
             indicators=[
                 dict(col_name="fast_ema", color="indianred", name="FAST EMA"),
                 dict(col_name="50_ema", color="indianred", name="50 EMA"),
                 dict(col_name="200_ema", color="indianred", name="200 EMA")]):

    candle = go.Scatter(
        x=df['time'],
        y=df['close'],
        name="line")

    data = [candle]

    for item in indicators:
        if df.__contains__(item['col_name']):
            fsma = go.Scatter(
                x=df['time'],
                y=df[item['col_name']],
                name=item['name'],
                line=dict(color=(item['color'])))
            data.append(fsma)

    if buy_signals:
        buys = go.Scatter(
            x=[item[0] for item in buy_signals],
            y=[item[1] for item in buy_signals],
            name="Buy Signals",
            mode="markers",
            marker_size=20
        )
        data.append(buys)

    if sell_signals:
        sells = go.Scatter(
            x=[item[0] for item in sell_signals],
            y=[item[1] for item in sell_signals],
            name="Sell Signals",
            mode="markers",
            marker_size=20
        )
        data.append(sells)

 
    layout = go.Layout(
        title=plot_title,
        xaxis={
            "title": plot_title,
            "rangeslider": {"visible": False},
            "type": "date"
        },
        yaxis={
            "fixedrange": False,
        })

    if trends is not False:
        layout['shapes'] = trends


    print(data)

    fig = go.Figure( go.Scatter(
        x=df['date'],
        y=df['close'],), layout = layout)

    plot(fig, filename='graphs/' + plot_title + '.html')
