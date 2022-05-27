import plotly.express as px


def plot_line_chart(df_plot, x, y, title, color, xaxis_title=None, yaxis_title=None):
    chart = px.line(df_plot, x=x, y=y, color=color, title=title)

    if xaxis_title and yaxis_title:
        chart.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    chart.show()


def plot_distance_charts(df):
    px.histogram(df, x="distance", nbins=None).show()

    box_plot = px.box(df, x="label", y="distance")

    box_plot.update_layout(yaxis_title="Distância", xaxis_title="K")

    box_plot.show()
