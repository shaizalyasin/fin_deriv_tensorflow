import dash
import numpy as np
import yfinance as yf
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from plotly.subplots import make_subplots
from pricer import BlackScholesPricer, BinomialTreePricer, MonteCarloPricer
from bond_pricer import VasicekModel, vasicek_calibration, HullWhiteModel, hull_white_calibration
from contract import EuropeanOptionContract
from model import MarketModel
from params import Params, TreeParams, MCParams
from enums import PutCall
from utils import calculate_time_to_expiry


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], suppress_callback_exceptions=True)
server = app.server


def option_input_form(pricer_id):
    card_body_content = [
        dbc.Row([
            dbc.Col(dbc.Label("Spot Price", className='input-form-label'), width=5),
            dbc.Col(dbc.Input(id=f'{pricer_id}_spot_price', type='number', placeholder='Enter Spot Price',
                              value=100, className='input-form-input'), width=6)
        ], className='mb-2'),

        dbc.Row([
            dbc.Col(dbc.Label("Strike Price", className='input-form-label'), width=5),
            dbc.Col(dbc.Input(id=f'{pricer_id}_strike_price', type='number', placeholder='Enter Strike Price',
                              value=90, className='input-form-input'), width=6)
        ], className='mb-2'),

        dbc.Row([
            dbc.Col(dbc.Label("Volatility (%)", className='input-form-label'), width=5),
            dbc.Col(dbc.Input(id=f'{pricer_id}_volatility', type='number', placeholder='Enter Volatility (%)',
                              value=25, className='input-form-input'), width=6)
        ], className='mb-2'),

        dbc.Row([
            dbc.Col(dbc.Label("Risk-Free Rate (%)", className='input-form-label'), width=5),
            dbc.Col(dbc.Input(id=f'{pricer_id}_risk_free_rate', type='number', placeholder='Enter Risk-Free Rate (%)',
                              value=5, className='input-form-input'), width=6)
        ], className='mb-2'),

        dbc.Row([
            dbc.Col(dbc.Label("Expiry Date", className='input-form-label'), width=5),
            dbc.Col(dcc.DatePickerSingle(id=f'{pricer_id}_time_to_expiry', placeholder='Enter Expiry Date',
                                         date='2025-05-08'), width=6)
        ], className='mb-2'),

        dbc.Row([
            dbc.Col(dbc.Label("Option Type", className='input-form-label'), width=5),
            dbc.Col(dbc.RadioItems(id=f'{pricer_id}_option_type',
                                   options=[{'label': 'Call', 'value': 'CALL'}, {'label': 'Put', 'value': 'PUT'}],
                                   value='CALL', inline=True, className='radio-items'), width=6)
        ], className='mb-2')
    ]
    if pricer_id == "bt":
        card_body_content.append(
            dbc.Row([
                dbc.Col(dbc.Label("Tree Steps (max. 10 displayed)", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_num_steps', type='number', placeholder='Number of Steps',
                                  value=10, className='input-form-input'), width=6)
            ], className='mb-2')
        )

    if pricer_id == "mc":
        card_body_content.append(
            dbc.Row([
                dbc.Col(dbc.Label("Simulations", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_num_paths', type='number', placeholder='Number of Paths',
                                  value=500, className='input-form-input'), width=6)
            ], className='mb-2')
        )
        card_body_content.append(
            dbc.Row([
                dbc.Col(dbc.Label("Time Steps", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_time_steps', type='number', placeholder='Number of Steps',
                                  value=50, className='input-form-input'), width=6)
            ], className='mb-2')
        )

    if pricer_id == "vasicek" or pricer_id == "hull_white":
        card_body_content = [
            dbc.Row([
                dbc.Col(dbc.Label("Bond Face Value ($)", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_face_value', type='number', placeholder='Enter Face Value',
                                  value=1000, className='input-form-input'), width=6)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(dbc.Label("Rate/Yield (%)", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_rate', type='number', placeholder='Enter Rate/Yield (%)',
                                  value=5, className='input-form-input'), width=6)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(dbc.Label("Time To Maturity (Years)", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_maturity', type='number', placeholder='Enter Maturity',
                                  value=5, className='input-form-input'), width=6)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(dbc.Label("Simulations", className='input-form-label'), width=5),
                dbc.Col(dbc.Input(id=f'{pricer_id}_num_paths', type='number', placeholder='Number of Simulations',
                                  value=1000, className='input-form-input'), width=6)
            ], className='mb-2')
        ]

    return dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H4("Input Parameters")),
            dbc.CardBody(card_body_content),
            dbc.Button("Calculate", id=f'{pricer_id}_calculate', color='primary', className='mt-2',
                       n_clicks=0)], className='input-form-container')
    ], width=4)


app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Financial Derivatives Calculator", className='main-title'))]),
    dbc.Row([
        dbc.Button("Black Scholes", id="btn_black_scholes", className="pricer-button black-scholes"),
        dbc.Button("Binomial Tree", id="btn_binomial_tree", className="pricer-button binomial-tree"),
        dbc.Button("Monte Carlo", id="btn_monte_carlo", className="pricer-button monte-carlo"),
        dbc.Button("Vasicek", id="btn_vasicek", className="pricer-button vasicek"),
        dbc.Button("Hull-White", id="btn_hull_white", className="pricer-button hull-white"),
    ], className="d-flex justify-content-center mt-4"),
    html.Div(id='pricer_content', className='mt-4'),
    html.Hr(),
    dbc.Row([dbc.Col(html.Footer(html.P("© Option Pricing Inc. All Rights Reserved.",
                                        className='text-center'), className='text-muted'), className='mt-2')]),
    dcc.Store(id='bs_store'),
    dcc.Store(id='bt_store'),
    dcc.Store(id='mc_store')
], fluid=True)


# Render different pricer forms based on button clicks
@app.callback(
    Output('pricer_content', 'children'),
    [Input('btn_black_scholes', 'n_clicks'),
     Input('btn_binomial_tree', 'n_clicks'),
     Input('btn_monte_carlo', 'n_clicks'),
     Input('btn_vasicek', 'n_clicks'),
     Input('btn_hull_white', 'n_clicks')],
    [State('bs_store', 'data'),
     State('bt_store', 'data'),
     State('mc_store', 'data')]
)
def render_pricer_content(btn_black_scholes, btn_binomial_tree, btn_monte_carlo, btn_vasicek, btn_hull_white,
                          bs_data, bt_data, mc_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div("Select a pricer model to get started.", className='text-center')
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'btn_black_scholes':
            return html.Div([
                html.H3("Black-Scholes Pricer", className='text-primary black-scholes-header'),
                dbc.Row([
                    option_input_form('bs'),
                    dbc.Col([
                        html.Div(id='bs_results', className='mt-4'),
                        html.Hr(),
                        html.Div(id='bs_custom_plot'),
                    ], width=8, className='results-plot-column')
                ])
            ], className='black-scholes-container')

        elif button_id == 'btn_binomial_tree':
            return html.Div([
                html.H3("Binomial Tree Pricer", className='text-primary binomial-tree-header'),
                dbc.Row([
                    option_input_form('bt'),
                    dbc.Col([
                        html.Div(id='bt_results', className='mt-4'),
                        html.Hr(),
                        html.Div(id='bt_custom_plot'),
                    ], width=8, className='results-plot-column')
                ])
            ], className='binomial-tree-container')

        elif button_id == 'btn_monte_carlo':
            return html.Div([
                html.H3("Monte Carlo Pricer", className='text-primary monte-carlo-header'),
                dbc.Row([
                    option_input_form('mc'),
                    dbc.Col([
                        html.Div(id='mc_results', className='mt-4'),
                        html.Hr(),
                        html.Div(id='mc_custom_plot'),
                    ], width=8, className='results-plot-column')
                ])
            ], className='monte-carlo-container')

        elif button_id == 'btn_vasicek':
            return html.Div([
                html.H3("Zero Coupon Bond - Vasicek", className='text-primary monte-carlo-header'),
                dbc.Row([
                    option_input_form('vasicek'),
                    dbc.Col([
                        html.Div(id='vasicek_results', className='mt-4'),
                        html.Hr(),
                        html.Div(id='vasicek_custom_plot'),
                    ], width=8, className='results-plot-column')
                ])
            ], className='monte-carlo-container')
        elif button_id == 'btn_hull_white':
            return html.Div([
                html.H3("Zero Coupon Bond - Hull White", className='text-primary monte-carlo-header'),
                dbc.Row([
                    option_input_form('hull_white'),
                    dbc.Col([
                        html.Div(id='hull_white_results', className='mt-4'),
                        html.Hr(),
                        html.Div(id='hull_white_custom_plot'),
                    ], width=8, className='results-plot-column')
                ])
            ], className='monte-carlo-container')


# Callback to calculate and display Black-Scholes results
@app.callback(
    [
        Output('bs_results', 'children'),
        Output('bs_custom_plot', 'children')
    ],
    [
        Input('bs_calculate', 'n_clicks'),
        State('bs_spot_price', 'value'),
        State('bs_strike_price', 'value'),
        State('bs_volatility', 'value'),
        State('bs_risk_free_rate', 'value'),
        State('bs_time_to_expiry', 'date'),
        State('bs_option_type', 'value')
    ]
)
def update_black_scholes_results(n_clicks, spot, strike, vol, rfr, expiry, opt_type):
    if n_clicks > 0 and all([spot, strike, vol, rfr, expiry]):
        t_expiry = calculate_time_to_expiry(expiry)

        market_model = MarketModel(spot=spot, risk_free_rate=rfr, sigma=vol)
        contract = EuropeanOptionContract(strike=strike, expiry=t_expiry,
                                          derivative_type=PutCall.CALL if opt_type == 'CALL' else PutCall.PUT)
        params = Params()

        pricer = BlackScholesPricer(contract=contract, model=market_model, params=params)

        price = pricer.calc_fair_value().numpy()
        delta = pricer.calc_delta().numpy()
        gamma = pricer.calc_gamma().numpy()
        vega = pricer.calc_vega().numpy()
        theta = pricer.calc_theta().numpy()
        rho = pricer.calc_rho().numpy()

        results_table = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Results", className='card-header-custom')),
                dbc.CardBody([dbc.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Option Price"), html.Td(f"{price:.2f}")]),
                        html.Tr([html.Td("Delta"), html.Td(f"{delta:.4f}")]),
                        html.Tr([html.Td("Gamma"), html.Td(f"{gamma:.4f}")]),
                        html.Tr([html.Td("Vega"), html.Td(f"{vega:.4f}")]),
                        html.Tr([html.Td("Theta"), html.Td(f"{theta:.4f}")]),
                        html.Tr([html.Td("Rho"), html.Td(f"{rho:.4f}")])
                    ])
                ], bordered=True, striped=True, hover=True)
                ])
            ])
        ])

        plot_options = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Plot Settings", className='card-header-custom')),
                dbc.CardBody([
                    dbc.Label("X-Axis", className='graph-options-label'),
                    dcc.Dropdown(
                        id='bs_x_axis',
                        options=[
                            {'label': 'Spot Price', 'value': 'spot'},
                            {'label': 'Strike Price', 'value': 'strike'},
                            {'label': 'Volatility', 'value': 'volatility'},
                            {'label': 'Risk-Free Rate', 'value': 'risk_free_rate'},
                            {'label': 'Time to Expiry', 'value': 'expiry'}
                        ],
                        value='spot',
                        clearable=False,
                        className='graph-options-dropdown'
                    ),
                    dbc.Label("Y-Axis", className='graph-options-label'),
                    dcc.Checklist(
                        id='bs_y_axis_metrics',
                        options=[
                            {'label': 'Option Price', 'value': 'price'},
                            {'label': 'Delta', 'value': 'delta'},
                            {'label': 'Gamma', 'value': 'gamma'},
                            {'label': 'Vega', 'value': 'vega'},
                            {'label': 'Theta', 'value': 'theta'},
                            {'label': 'Rho', 'value': 'rho'}
                        ],
                        value=['price'],
                        inline=True,
                        className='graph-options-checklist'
                    ),
                    dcc.Graph(id='bs_custom_graph', className='mt-4')
                ], className='graph-options-card')
            ])
        ])

        return results_table, plot_options
    return html.Div(), html.Div()


@app.callback(
    Output('bs_custom_graph', 'figure'),
    [
        Input('bs_x_axis', 'value'),
        Input('bs_y_axis_metrics', 'value'),
        State('bs_spot_price', 'value'),
        State('bs_strike_price', 'value'),
        State('bs_volatility', 'value'),
        State('bs_risk_free_rate', 'value'),
        State('bs_time_to_expiry', 'date'),
        State('bs_option_type', 'value')
    ]
)
def update_bs_custom_plot(x_axis, y_metrics, spot, strike, vol, rfr, expiry, opt_type):
    if all([spot, strike, vol, rfr, expiry, y_metrics]):
        t_expiry = calculate_time_to_expiry(expiry)
        market_model = MarketModel(spot=spot, risk_free_rate=rfr, sigma=vol)
        contract = EuropeanOptionContract(strike=strike, expiry=t_expiry,
                                          derivative_type=PutCall.CALL if opt_type == 'CALL' else PutCall.PUT)
        params = Params()
        pricer = BlackScholesPricer(contract=contract, model=market_model, params=params)

        range_vals = {
            'spot': [spot * (0.5 + 0.01 * i) for i in range(100)],
            'strike': [strike * (0.5 + 0.01 * i) for i in range(100)],
            'volatility': [vol * (0.5 + 0.01 * i) for i in range(100)],
            'risk_free_rate': [rfr * (0.5 + 0.01 * i) for i in range(100)],
            'expiry': [t_expiry * (0.5 + 0.01 * i) for i in range(100)]
        }

        plot_fig = make_subplots(specs=[[{"secondary_y": True}]])
        plot_fig.update_layout(template='plotly_white')

        for metric in y_metrics:
            metric_values = []
            for val in range_vals[x_axis]:
                if x_axis == 'spot':
                    market_model.spot = val
                elif x_axis == 'strike':
                    contract.strike = val
                elif x_axis == 'volatility':
                    market_model.sigma = val
                elif x_axis == 'risk_free_rate':
                    market_model.risk_free_rate = val
                elif x_axis == 'expiry':
                    contract.expiry = val

                if metric == 'price':
                    calculated_value = pricer.calc_fair_value().numpy()
                else:
                    calculated_value = getattr(pricer, f'calc_{metric}')().numpy()

                metric_values.append(calculated_value)

            secondary_y = True if metric in ['theta', 'gamma'] else False

            plot_fig.add_trace(
                go.Scatter(x=range_vals[x_axis], y=metric_values, mode='lines', name=metric),
                secondary_y=secondary_y,
            )

        plot_fig.update_layout(title=f"Metrics vs {x_axis.capitalize()}")
        plot_fig.update_yaxes(title_text="Primary Axis Metrics", secondary_y=False)
        plot_fig.update_yaxes(title_text="Secondary Axis Metrics", secondary_y=True)

        return plot_fig

    return go.Figure(layout=go.Layout(template='plotly_white', width=800, height=600, autosize=True))


# Callback to calculate and display Binomial Tree results
@app.callback(
    [
        Output('bt_results', 'children'),
        Output('bt_custom_plot', 'children')
    ],
    [
        Input('bt_calculate', 'n_clicks'),
        State('bt_spot_price', 'value'),
        State('bt_strike_price', 'value'),
        State('bt_volatility', 'value'),
        State('bt_risk_free_rate', 'value'),
        State('bt_time_to_expiry', 'date'),
        State('bt_option_type', 'value'),
        State('bt_num_steps', 'value')
    ]
)
def update_binomial_tree_results(n_clicks, spot, strike, vol, rfr, expiry, opt_type, steps):
    if n_clicks > 0 and all([spot, strike, vol, rfr, expiry, steps]):
        t_expiry = calculate_time_to_expiry(expiry)

        market_model = MarketModel(spot=spot, risk_free_rate=rfr, sigma=vol)
        contract = EuropeanOptionContract(strike=strike, expiry=t_expiry, derivative_type=PutCall.CALL if opt_type == 'CALL' else PutCall.PUT)
        tree_params = TreeParams(num_steps=steps)

        pricer = BinomialTreePricer(contract=contract, model=market_model, params=tree_params)

        price = pricer.calc_fair_value()
        pricer.build_tree()

        stock_tree = pricer.stock_tree
        option_tree = pricer.option_tree

        delta = pricer.calc_delta()
        gamma = pricer.calc_gamma()
        vega = pricer.calc_vega()
        theta = pricer.calc_theta()
        rho = pricer.calc_rho()

        results_table = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Results", className='card-header-custom')),
                dbc.CardBody([dbc.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Option Price"), html.Td(f"{price:.2f}")]),
                        html.Tr([html.Td("Delta"), html.Td(f"{delta:.4f}")]),
                        html.Tr([html.Td("Gamma"), html.Td(f"{gamma:.4f}")]),
                        html.Tr([html.Td("Vega"), html.Td(f"{vega:.4f}")]),
                        html.Tr([html.Td("Theta"), html.Td(f"{theta:.4f}")]),
                        html.Tr([html.Td("Rho"), html.Td(f"{rho:.4f}")])
                    ])
                ], bordered=True, striped=True, hover=True)
                ])
            ])
        ])

        tree_plot = create_binomial_tree_plot(stock_tree, option_tree)

        return results_table, tree_plot
    return html.Div(), html.Div()


def create_tree_trace(tree, title):
    traces = []
    annotations = []
    steps = tree.shape[1] - 1
    if steps > 10:
        steps = 10

    x_positions = {}
    y_positions = {}

    def calculate_positions():
        index = 0
        for level in range(steps + 1):
            for position in range(level + 1):
                x_positions[index] = level
                y_positions[index] = position - (level / 2.0)
                index += 1

    calculate_positions()

    offset = 0.25

    for level in range(steps + 1):
        for position in range(level + 1):
            node_index = position + (level * (level + 1)) // 2
            x = x_positions.get(node_index)
            y = y_positions.get(node_index)

            if x is None or y is None:
                continue

            if level < steps:
                left_child_index = node_index + level + 1
                right_child_index = node_index + level + 2
                if left_child_index in x_positions:
                    next_x_left = x_positions[left_child_index]
                    next_y_left = y_positions[left_child_index]
                    dx = next_x_left - x
                    dy = next_y_left - y
                    norm = (dx ** 2 + dy ** 2) ** 0.5
                    annotations.append(dict(
                        ax=x + dx * offset / norm,
                        ay=y + dy * offset / norm,
                        axref='x', ayref='y',
                        x=next_x_left - dx * offset / norm,
                        y=next_y_left - dy * offset / norm,
                        xref='x', yref='y',
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='black'
                    ))

                if right_child_index in x_positions:
                    next_x_right = x_positions[right_child_index]
                    next_y_right = y_positions[right_child_index]
                    dx = next_x_right - x
                    dy = next_y_right - y
                    norm = (dx ** 2 + dy ** 2) ** 0.5
                    annotations.append(dict(
                        ax=x + dx * offset / norm,
                        ay=y + dy * offset / norm,
                        axref='x', ayref='y',
                        x=next_x_right - dx * offset / norm,
                        y=next_y_right - dy * offset / norm,
                        xref='x', yref='y',
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='black'
                    ))

            value = tree[position, level]
            traces.append(go.Scatter(
                x=[x],
                y=[y],
                text=[f"{value:.2f}"],
                mode='text+markers',
                textposition='middle center',
                marker=dict(size=30, symbol='square', color='rgba(255, 255, 0, 0.4)'),
                textfont=dict(size=10, color='black'),
                showlegend=False
            ))

    layout = go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        plot_bgcolor='white',
        autosize=True,
        margin=dict(t=5, b=5, l=5, r=5),
        annotations=annotations
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def create_binomial_tree_plot(stock_tree, option_tree):
    option_trace = create_tree_trace(option_tree, "Option Price Tree")
    underlying_trace = create_tree_trace(stock_tree, "Underlying Price Tree")

    option_graph = dcc.Graph(figure=option_trace)
    underlying_graph = dcc.Graph(figure=underlying_trace)

    return html.Div([
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Option Price Tree", className='card-header-custom')),
                dbc.CardBody(option_graph, className='plot-card-body')
            ], className='plot-card')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Underlying Price Tree", className='card-header-custom')),
                dbc.CardBody(underlying_graph, className='plot-card-body')
            ], className='plot-card')
        ])
    ])


# Callback to calculate and display Monte Carlo results
@app.callback(
    [
        Output('mc_results', 'children'),
        Output('mc_custom_plot', 'children')
    ],
    [
        Input('mc_calculate', 'n_clicks'),
        State('mc_spot_price', 'value'),
        State('mc_strike_price', 'value'),
        State('mc_volatility', 'value'),
        State('mc_risk_free_rate', 'value'),
        State('mc_time_to_expiry', 'date'),
        State('mc_option_type', 'value'),
        State('mc_num_paths', 'value'),
        State('mc_time_steps', 'value')
    ]
)
def update_monte_carlo_results(n_clicks, spot, strike, vol, rfr, expiry, opt_type, paths, steps):
    if n_clicks and all([spot, strike, vol, rfr, expiry, paths, steps]):
        t_expiry = calculate_time_to_expiry(expiry)

        market_model = MarketModel(spot=spot, risk_free_rate=rfr, sigma=vol)
        option_type = PutCall.CALL if opt_type == 'CALL' else PutCall.PUT
        contract = EuropeanOptionContract(strike=strike, expiry=t_expiry, derivative_type=option_type)
        mc_params = MCParams(num_paths=paths, time_steps=steps)
        pricer = MonteCarloPricer(contract=contract, model=market_model, params=mc_params, save_paths=True)

        price = pricer.calc_fair_value()
        delta = pricer.calc_delta()
        gamma = pricer.calc_gamma()
        vega = pricer.calc_vega()
        theta = pricer.calc_theta()
        rho = pricer.calc_rho()

        results_table = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Results", className='card-header-custom')),
                dbc.CardBody([dbc.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Option Price"), html.Td(f"{price:.2f}")]),
                        html.Tr([html.Td("Delta"), html.Td(f"{delta:.4f}")]),
                        html.Tr([html.Td("Gamma"), html.Td(f"{gamma:.4f}")]),
                        html.Tr([html.Td("Vega"), html.Td(f"{vega:.4f}")]),
                        html.Tr([html.Td("Theta"), html.Td(f"{theta:.4f}")]),
                        html.Tr([html.Td("Rho"), html.Td(f"{rho:.4f}")])
                    ])
                ], bordered=True, striped=True, hover=True)])
            ], className='plot-card', style={'margin-bottom': '20px'})
        ])

        plot_fig = create_monte_carlo_plot(pricer)
        monte_graph = dcc.Graph(figure=plot_fig, style={'width': '100%', 'height': '100%'})
        plot_div = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Monte Carlo Simulation for Stock Price", className='card-header-custom')),
                dbc.CardBody([monte_graph], className='plot-card-body')
            ], className='plot-card', style={'width': '100%', 'max-width': '1000px', 'margin': 'auto'})
        ])

        return results_table, plot_div
    return html.Div(), html.Div()


def create_monte_carlo_plot(pricer):

    paths = pricer.get_paths()
    if paths is None:
        raise ValueError("No paths saved. Ensure that `save_paths` is True when initializing the pricer.")

    # Calculate mean and confidence intervals for the paths
    mean_path = np.mean(paths, axis=1)
    upper_bound = np.percentile(paths, 97.5, axis=1)
    lower_bound = np.percentile(paths, 2.5, axis=1)

    fig = go.Figure()

    for path in paths.T:
        fig.add_trace(go.Scatter(
            y=path,
            mode='lines',
            line=dict(width=1),
            opacity=0.3,
            showlegend=False
        ))

    # Add mean path
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        name='Mean',
        line=dict(color='red', width=2)
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        y=upper_bound,
        mode='lines',
        name='97.5% Percentile',
        line=dict(color='green', width=1, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        y=lower_bound,
        mode='lines',
        name='2.5% Percentile',
        line=dict(color='green', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0, 100, 80, 0.2)'
    ))

    fig.update_layout(
        xaxis_title='Number of Steps',
        yaxis_title='Stock Price',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        ),
        width=900,
        height=600,
        autosize=False
    )

    return fig


# Callback to calculate and display Vasicek results
@app.callback(
    [
        Output('vasicek_results', 'children'),
        Output('vasicek_custom_plot', 'children')
    ],
    [
        Input('vasicek_calculate', 'n_clicks'),
        State('vasicek_face_value', 'value'),
        State('vasicek_rate', 'value'),
        State('vasicek_maturity', 'value'),
        State('vasicek_num_paths', 'value')
    ]
)
def update_vasicek_results(n_clicks, face_value, rate, maturity, num_paths):
    if n_clicks > 0 and all([face_value, rate, maturity, num_paths]):
        # data
        data = yf.download("^TNX", start="2024-04-01", end="2024-05-01", interval="1d")
        tnx_rate = 0.01 * data['Adj Close'].dropna().values

        maturities = np.array([10] * len(tnx_rate))

        initial_guess = (0.1, 0.05, 0.01)

        # model calibration
        calibrated_params = vasicek_calibration(tnx_rate, maturities, initial_guess, epochs=100)
        alpha, beta, sigma = calibrated_params
        # print(f"Calibrated Parameters: alpha={alpha}, beta={beta}, sigma={sigma}")

        vasicek_model = VasicekModel(alpha, beta, sigma)
        current_rate = rate / 100.0

        analytical_price = vasicek_model.bond_price(face_value, current_rate, maturity)
        monte_carlo_price = vasicek_model.monte_carlo_bond_price(face_value, current_rate, maturity,
                                                                 num_paths=num_paths)
        bond_yield = vasicek_model.bond_yield(face_value, analytical_price.numpy(), maturity)

        analytical_price = analytical_price.numpy()
        bond_yield = bond_yield
        # print(analytical_price)
        # print(bond_yield)
        # print(monte_carlo_price)

        results_table = html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Results", className='card-header-custom')),
                dbc.CardBody([dbc.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Speed Of Mean Reversion"), html.Td(f"{alpha:.4f}")]),
                        html.Tr([html.Td("Long-term Mean Level"), html.Td(f"{beta:.4f}")]),
                        html.Tr([html.Td("Volatility"), html.Td(f"{sigma:.4f}")]),
                        html.Tr([html.Td("Analytical PDE Solution"), html.Td(f"{analytical_price:.2f}")]),
                        html.Tr([html.Td("Monte Carlo Estimation"), html.Td(f"{monte_carlo_price:.2f}")]),
                        html.Tr([html.Td("Bond Yield"), html.Td(f"{bond_yield:.4f}")])
                    ])
                ], bordered=True, striped=True, hover=True)
                ])
            ])
        ])

        # Plot for CBOE Interest Rate 10 Year T Note
        fig_rate_evolution = go.Figure(layout=go.Layout(template='plotly_white', autosize=True))
        fig_rate_evolution.add_trace(go.Scatter(x=data.index, y=tnx_rate, mode='lines', name='CBOE 10 Year T Note'))
        fig_rate_evolution.update_layout(title='CBOE Interest Rate 10 Year T Note Evolution',
                                         xaxis_title='Date', yaxis_title='Rate')

        # Plot for Bond Prices
        r0_values = np.linspace(0.01, 0.1, 10)
        mc_prices = [vasicek_model.monte_carlo_bond_price(face_value, r0, maturity, num_paths=num_paths)
                     for r0 in r0_values]
        analytical_prices = [vasicek_model.bond_price(face_value, r0, maturity).numpy() for r0 in r0_values]

        fig_bond_prices = go.Figure(layout=go.Layout(template='plotly_white', width=700, autosize=True))
        fig_bond_prices.add_trace(go.Scatter(x=r0_values, y=mc_prices, mode='lines+markers', name='Monte Carlo Price'))
        fig_bond_prices.add_trace(
            go.Scatter(x=r0_values, y=analytical_prices, mode='lines+markers', name='Analytical Price'))
        fig_bond_prices.update_layout(title='Bond Prices for Different Initial Rates (r0)',
                                      xaxis_title='Rate (r0)', yaxis_title='Price')

        # maturities_term = np.linspace(0.1, 30, 30)
        # analytical_prices_term = [vasicek_model.bond_price(face_value, rate / 100, T) for T in maturities_term]
        # yields = [100 * ((face_value / p) ** (1 / T) - 1) for p, T in zip(analytical_prices_term, maturities_term)]
        #
        # fig_term_structure = go.Figure(layout=go.Layout(template='plotly_white', autosize=True))
        # fig_term_structure.add_trace(
        #     go.Scatter(x=maturities_term, y=yields, mode='lines+markers', name='Yield Curve'))
        # fig_term_structure.update_layout(title='Term Structure of Interest Rates',
        #                                  xaxis_title='Maturity (Years)', yaxis_title='Yield (%)')

        rate_evolution = dcc.Graph(figure=fig_rate_evolution)
        bond_prices = dcc.Graph(figure=fig_bond_prices)
        # term_structure_graph = dcc.Graph(figure=fig_term_structure)

        graph_fig = html.Div([
            html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H4("CBOE Interest Rate 10 Year T Note Evolution",
                                           className='card-header-custom')),
                    dbc.CardBody(rate_evolution, className='plot-card-body')
                ], className='plot-card')
            ], style={'margin-bottom': '10px'}),
            html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H4("Bond Prices for Different Initial Rates (r0)",
                                           className='card-header-custom')),
                    dbc.CardBody(bond_prices, className='plot-card-body')
                ], className='plot-card')
            ]),
            # html.Div([
            #     dbc.Card([
            #         dbc.CardHeader(
            #             html.H4("Term Structure", className='card-header-custom')),
            #         dbc.CardBody(term_structure_graph, className='plot-card-body')
            #     ], className='plot-card')
            # ])
        ])

        return results_table, graph_fig
    return html.Div(), html.Div()


# Callback to calculate and display Hull-White Model results
@app.callback(
    [
        Output('hull_white_results', 'children'),
        Output('hull_white_custom_plot', 'children'),
    ],
    [
        Input('hull_white_calculate', 'n_clicks'),
        State('hull_white_face_value', 'value'),
        State('hull_white_rate', 'value'),
        State('hull_white_maturity', 'value'),
        State('hull_white_num_paths', 'value')
    ]
)
def update_hull_white_results(n_clicks, face_value, rate, maturity, num_paths):
    if n_clicks > 0 and all([face_value, rate, maturity, num_paths]):
        try:
            # print(f"Face Value: {face_value}, Rate: {rate}, Maturity: {maturity}, Num Paths: {num_paths}")
            # Data observed from US Federal Reserve System
            rates = np.array([5.50, 5.45, 5.82, 5.14, 4.82, 4.61, 4.43, 4.42, 4.41, 4.65, 4.55],
                             dtype=np.float32) / 100
            maturities = np.array([0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float32)

            # model calibration
            hull_white_model = hull_white_calibration(rates, maturities, epochs=1000)
            current_rate = rate / 100

            analytical_price = hull_white_model.get_zcb_price(face_value, current_rate, maturity)
            monte_carlo_price = hull_white_model.monte_carlo_zcb_price(face_value, current_rate, maturity,
                                                                       num_simulations=num_paths)
            bond_yield_value = hull_white_model.bond_yield(analytical_price, face_value, maturity)

            # print(f"Calibrated Parameters: Alpha={hull_white_model.a.numpy()}, "
            #       f"Sigma={hull_white_model.sigma.numpy()}, Theta={hull_white_model.theta.numpy()}")
            # print(f"Analytical Price: {analytical_price}")
            # print(f"Monte Carlo Price: {monte_carlo_price}")
            # print(f"Bond Yield: {bond_yield_value}")
            alpha = hull_white_model.a.numpy()
            sigma = hull_white_model.sigma.numpy()
            theta = hull_white_model.theta.numpy()

            results_table = html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H4("Results", className='card-header-custom')),
                    dbc.CardBody([dbc.Table([
                        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                        html.Tbody([
                            html.Tr([html.Td("Speed Of Mean Reversion"), html.Td(f"{alpha:.4f}")]),
                            html.Tr([html.Td("Volatility"), html.Td(f"{sigma:.4f}")]),
                            html.Tr([html.Td("Time-Dependent Drift"), html.Td(f"{theta}")]),
                            html.Tr([html.Td("Analytical PDE Solution"), html.Td(f"{analytical_price:.2f}")]),
                            html.Tr([html.Td("Monte Carlo Estimation"), html.Td(f"{monte_carlo_price:.2f}")]),
                            html.Tr([html.Td("Bond Yield"), html.Td(f"{bond_yield_value:.4f}")])
                        ])
                    ], bordered=True, striped=True, hover=True)
                    ])
                ])
            ])

            # Plot for Calibration Data
            fig_calibration_data = go.Figure(layout=go.Layout(template='plotly_white', autosize=True))
            fig_calibration_data.add_trace(
                go.Scatter(x=maturities, y=rates, mode='lines+markers', name='Calibration Data'))
            fig_calibration_data.update_layout(title='Calibration Data: Rates vs Maturities',
                                               xaxis_title='Maturity (Years)', yaxis_title='Rate')

            # Plot for Bond Prices
            r0_values = np.linspace(0.01, 0.1, 10)
            mc_prices = [hull_white_model.monte_carlo_zcb_price(face_value, r0, maturity, num_simulations=num_paths)
                         for r0 in r0_values]
            analytical_prices = [hull_white_model.get_zcb_price(face_value, r0, maturity) for r0 in r0_values]

            fig_bond_prices = go.Figure(layout=go.Layout(template='plotly_white', width=700, autosize=True))
            fig_bond_prices.add_trace(
                go.Scatter(x=r0_values, y=mc_prices, mode='lines+markers', name='Monte Carlo Price'))
            fig_bond_prices.add_trace(
                go.Scatter(x=r0_values, y=analytical_prices, mode='lines+markers', name='Analytical Price'))
            fig_bond_prices.update_layout(title='Bond Prices for Different Initial Rates (r0)',
                                          xaxis_title='Rate (r0)', yaxis_title='Price')

            # maturities_term = np.linspace(1, 30, 30)
            # analytical_prices_term = [hull_white_model.get_zcb_price(face_value, rate / 100, T) for T in maturities]
            # yields = [100 * ((face_value / p) ** (1 / T) - 1) for p, T in zip(analytical_prices_term, maturities)]
            #
            # fig_term_structure = go.Figure(layout=go.Layout(template='plotly_white', autosize=True))
            # fig_term_structure.add_trace(
            #     go.Scatter(x=maturities, y=yields, mode='lines+markers', name='Yield Curve'))
            # fig_term_structure.update_layout(title='Term Structure of Interest Rates',
            #                                  xaxis_title='Maturity (Years)', yaxis_title='Yield (%)')

            calibration_data_plot = dcc.Graph(figure=fig_calibration_data)
            bond_prices_plot = dcc.Graph(figure=fig_bond_prices)
            # term_structure_graph = dcc.Graph(figure=fig_term_structure)

            graph_fig = html.Div([
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("Federal Reserve Bank of St Louis", className='card-header-custom')),
                        dbc.CardBody(calibration_data_plot, className='plot-card-body')
                    ], className='plot-card')
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("Bond Prices for Different Initial Rates (r0)",
                                    className='card-header-custom')),
                        dbc.CardBody(bond_prices_plot, className='plot-card-body')
                    ], className='plot-card')
                ]),
                # html.Div([
                #     dbc.Card([
                #         dbc.CardHeader(
                #             html.H4("Term Structure", className='card-header-custom')),
                #         dbc.CardBody(term_structure_graph, className='plot-card-body')
                #     ], className='plot-card')
                # ])
            ])

            return results_table, graph_fig

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return html.Div(f"An error occurred: {str(e)}"), html.Div()

    return html.Div(), html.Div()


if __name__ == '__main__':
    app.run_server(debug=False)
