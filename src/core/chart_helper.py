from tkinter import font
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

    chart.update_layout(
        legend=dict(
            y=1,
            x=1
        )
    )

    # chart.update_layout(
    #     legend=dict(
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="right",
    #         x=1
    #     ),
    #     font=dict(size=18),
    # )

    # height=800,
    # legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=1
    # ),

    chart.show()


def plot_distribution_charts(df, distribution_collumn, short_title, long_title):
    histogram = px.histogram(df, x=distribution_collumn, nbins=None)

    histogram.update_layout(yaxis_title="Amount of Sentences", xaxis_title=short_title)

    histogram.show()

    box_plot = px.box(df, x="label", y=distribution_collumn)

    box_plot.update_layout(
        yaxis_title=short_title,
        xaxis_title="Cluster",
        title=long_title
    )
    
    box_plot.show()
