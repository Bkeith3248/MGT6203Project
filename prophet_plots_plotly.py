import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from IPython.display import display,Markdown
import math
import statistics as stats
import numpy as np
from datetime import datetime as dt
import warnings
warnings.simplefilter('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.dates import AutoDateLocator

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_forecast_component
from fbprophet.plot import seasonality_plot_df

def plot_model(model_data = None, model = None,forcast = None,fourier_order = None):

    temp_vis = forcast.copy()
    temp_vis['y'] = model_data['y']
    y_temp = temp_vis['y']
    x_temp =  temp_vis['ds']
    y_temp_hat = temp_vis['yhat']
    y_temp_hat_lower = temp_vis['yhat_lower']
    y_temp_hat_upper =temp_vis['yhat_upper']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=x_temp, y=y_temp,
                            marker=dict(color='#000000',size = 5),
                            line = dict(color='#000000',width = 1),
                            mode='markers+lines',
                            name='Late Fees Collected (Actual)'),secondary_y=False,)

    fig.add_trace(go.Scatter(x=x_temp, y=y_temp_hat,
                            line = dict(color='#FF3333',width = 1),
                            mode='lines',
                            name='Late Fees Collected (Forcast)'),secondary_y=False,)

    fig.add_trace(go.Scatter(x=x_temp, y=y_temp_hat_lower,
                            line = dict(color='#bf9fa2',width = 1),
                            mode='lines',
                            name='Lower Forcast Range'),secondary_y=False,)

    fig.add_trace(go.Scatter(x=x_temp, y=y_temp_hat_upper,
                            line = dict(color='#bf9fa2',width = 1),
                            mode='lines',
                            fill='tonexty',
                            name='Upper Forcast Range'),secondary_y=False,)

    fig.update_layout(legend=dict(orientation = 'h',yanchor="bottom",
                                    y=1.05,
                                    xanchor="left",
                                    x=0.01))
    
    fig.update_layout(title={'text': f'Fourier Order =  {fourier_order}',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    
    fig.update_layout(hovermode="x unified",clickmode ='select')

    fig.show()

    
    #fig = model.plot(forcast, xlabel='Date', ylabel='Late Fees Collected (Order: {})'.format(i))
    
    #a = add_changepoints_to_plot(fig.gca(), model, forcast)
    #fig.show()
    #fig = model.plot_components(forcast)
    #fig.show()
    
    #fig = plot_forecast_component(model,forcast,'trend')

def plot_model_components(model =  None,forcast = None):
    components = ['trend']
    if model.train_holiday_names is not None and 'holidays' in forcast:
        components.append('holidays')
    # Plot weekly seasonality, if present
    if 'weekly' in model.seasonalities and 'weekly' in forcast:
        components.append('weekly')
    # Yearly if present
    if 'yearly' in model.seasonalities and 'yearly' in forcast:
        components.append('yearly')
    # Other seasonalities
    components.extend([
        name for name in sorted(model.seasonalities)
        if name in forcast and name not in ['weekly', 'yearly']
    ])


    regressors = {'additive': False, 'multiplicative': False}
    for name, props in model.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in forcast:
            components.append('extra_regressors_{}'.format(mode))
            
    #plot_components
    fig = make_subplots(rows = len(components),
                        cols = 1,
                        subplot_titles = (components),
                        horizontal_spacing = 0.05,
                        vertical_spacing = 0.15)

    row_count = 1
    for component in components:
        if component == 'trend':
            fig.append_trace(go.Scatter(
                x=forcast['ds'],
                y=forcast[component],
                name = 'Trend',
                line = dict(color='#940303',width = 3)
                ), row=row_count, col=1)
        elif component in model.seasonalities:
            #plot_weekly
            if component == 'weekly' or model.seasonalities[component]['period'] == 7:
                days = (pd.date_range(start='2017-01-01', periods=7) + pd.Timedelta(days=0))
                df_w = seasonality_plot_df(model, days)
                seas = model.predict_seasonal_components(df_w)
                days = days.strftime("%A")
                fig.append_trace(go.Scatter(
                    x=days,
                    y=seas[component],
                    name = 'Weekly Seasonality',
                    line = dict(color='#940303',width = 2)
                    ), row=row_count, col=1)
                if model.seasonalities[component]['mode'] == 'multiplicative':  
                    fig.update_yaxes(tickformat=".2%",row=row_count, col=1)       
            elif component == 'yearly' or model.seasonalities[component]['period'] == 365.25:
                days = (pd.date_range(start='2017-01-01', periods=365) + pd.Timedelta(days=0))
                df_y = seasonality_plot_df(model, days)
                seas = model.predict_seasonal_components(df_y)
                fig.append_trace(go.Scatter(
                    x=df_y['ds'].dt.to_pydatetime(),
                    y=seas[component],
                    name = 'Yearly Seasonality',
                    line = dict(color='#940303',width = 2)
                    ), row=row_count, col=1)
                if model.seasonalities[component]['mode'] == 'multiplicative':  
                    fig.update_yaxes(tickformat=".2%",row=row_count, col=1)       
            else:
                start = pd.to_datetime('2017-01-01 0000')
                period = model.seasonalities[component]['period']
                end = start + pd.Timedelta(days=period)
                plot_points = 200
                days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
                df_y = seasonality_plot_df(model, days)
                seas = model.predict_seasonal_components(df_y)
                fig.append_trace(go.Scatter(
                    x=df_y['ds'].dt.to_pydatetime(),
                    y=seas[component],
                    name = 'Other Seasonality',
                    line = dict(color='#940303',width = 2)
                    ), row=row_count, col=1)
                if model.seasonalities[component]['mode'] == 'multiplicative':  
                    fig.update_yaxes(tickformat=".2%",row=row_count, col=1)
                
        #elif component in ['holidays','extra_regressors_additive','extra_regressors_multiplicative',]:

        #    if model.seasonalities[component]['mode'] == 'multiplicative':  
        #        fig.update_yaxes(tickformat=".2%",row=row_count, col=1)
                
                    
        row_count += 1
    fig.update_layout(showlegend=False)
    fig.update_layout(margin = dict(l=10,r=10, b=10, t=50 ))
    fig.show()


    
