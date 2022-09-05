import plotly.express as px


def plot_line_chart(df_plot, x, y, title, color, xaxis_title=None, yaxis_title=None):
    chart = px.line(df_plot, x=x, y=y, color=color, title=title)

    if xaxis_title and yaxis_title:
        chart.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    # chart.update_layout(
    #                   paper_bgcolor='rgba(0,0,0,0)',
    #                   plot_bgcolor='rgba(0,0,0,0)'
    #                   )
    # chart.update_layout(plot_bgcolor='#fff')
    chart.update_traces(textposition="bottom right")

    chart.show()


def plot_distance_charts(df):
    histogram = px.histogram(df, x="distance", nbins=None)

    histogram.update_layout(yaxis_title="Amount of Sentences", xaxis_title="Distance")

    histogram.show()

    box_plot = px.box(df, x="label", y="distance")

    box_plot.update_layout(yaxis_title="Distance", xaxis_title="Cluster")

    box_plot.show()
