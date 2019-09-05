import numpy as np

from scipy.stats import norm, gamma, beta
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from .data_simulator import SimulateBiGMM


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Panel
from bokeh.layouts import column, row, widgetbox, layout
from bokeh.models.widgets import (
    Slider,
    Button,
    Div,
    TextInput,
    Select,
    TextAreaInput,
    CheckboxButtonGroup,
    RangeSlider,
)


from bokeh.models import CustomJS, Paragraph


from .js_3d_bokeh import Surface3d

from .mdn import MDN_network, mdn_loss_fn, train_epoch, simulate_condprob_trained

import torch


def style(p):
    # Title
    p.title.align = "center"
    p.title.text_font_size = "20pt"
    p.title.text_font = "serif"

    # Axis titles
    p.xaxis.axis_label_text_font_size = "14pt"
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_style = "bold"

    # Tick labels
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"

    return p


def nn():
    ###------------------------PARAMETER DEFAULTS-----------------------------###
    # range[Lower, Upper, Step Size]
    ### - SAMPLING Parameters
    d_nsamp, r_nsamp = 500, [100, 2000, 50]  # Number of samples

    plot_data = figure(
        plot_height=400,
        plot_width=400,
        title="Simulated points (x,y)",
        toolbar_location="above",
        x_axis_label="x",
        y_axis_label="y",
        tools="pan,save,box_zoom,wheel_zoom",
    )

    source = ColumnDataSource(data=dict(x=[], y=[], z=[]))
    source.data = dict(x=np.zeros([1, 1]), y=np.zeros([1, 1]), z=np.zeros([1, 1]))
    surface_title = Div(text="<b> Surafce plot of the joint distribution z = p(x,y) </b>")
    surface = Surface3d(x="x", y="y", z="z", data_source=source)

    plot_conditional = figure(
        plot_height=400,
        plot_width=800,
        title="Conditional density p(y/x=X)",
        toolbar_location="above",
        x_axis_label="y",
        y_axis_label="CDF",
        tools="pan,save,box_zoom,wheel_zoom",
    )

    style(plot_conditional)

    plot_clear = Button(label="Clear All", button_type="warning")

    # Simulation of GMM

    ctl_title = Div(text="<h3>Simulator</h3>")
    div1 = Div(
        text="""<p style="border:3px; border-style:solid; border-color:grey; padding: 1em;">
                    Hyperparameters of bi-GMM components (refer to the blog post for details)<br />  
                    - n_kernels: Nb of componenets<br />  
                    - means_loc: mean of a normal distribution from where the means of components are sampled. <br /> 
                    - means_scale: associated scale. <br /> 
                    - covariance_x_loc: mean of a normal distribution from where the covariances of X are simpled <br /> 
                    - covariance_x_scale: associated scale. <br /> 
                    - covariance_y_loc: mean of a normal distribution from where the covariances of Y are simpled <br /> 
                    - covariance_y_scale: associated scale. <br /> 
                    </p>""",
        width=300,
        height=330,
    )

    ctl_nsamp = Slider(
        title="Number of samples",
        value=d_nsamp,
        start=r_nsamp[0],
        end=r_nsamp[1],
        step=r_nsamp[2],
    )

    n_kernels = Slider(title="Number of mixtures", value=5, start=1, end=10, step=1)

    means_loc = TextInput(title="means_loc", value="0.0")
    means_scale = TextInput(title="means_scale", value="4.0")

    covariance_x_loc = TextInput(title="covariance_x_loc", value="1.0")
    covariance_x_scale = TextInput(title="covariance_x_scale", value="0.5")

    covariance_y_loc = TextInput(title="covariance_y_loc", value="1.0")
    covariance_y_scale = TextInput(title="covariance_y_scale", value="0.5")

    plot_sim = Button(label="Simulate", button_type="primary")

    simulate = widgetbox(
        ctl_title,
        div1,
        ctl_nsamp,
        plot_sim,
        n_kernels,
        means_loc,
        means_scale,
        covariance_x_loc,
        covariance_x_scale,
        covariance_y_loc,
        covariance_y_scale,
    )

    ### MDN

    mdn_title = Div(text="<h3>MDN Training </h3>")
    mdn_text = Div(text="Choose training parameters")

    n_gaussians = Slider(
        title="Number of mixtures to fit", value=10, start=1, end=20, step=1
    )

    n_layers = Slider(title="Number of hidden layers", value=1, start=1, end=3, step=1)

    n_hidden = Slider(
        title="Size of hidden neurons", value=20, start=5, end=10, step=1
    )

    dropout = Slider(
        title="Dropout regularization", value=0.05, start=0, end=0.2, step=0.01
    )

    l2 = Slider(title="L2 regularization", value=0.001, start=0, end=0.1, step=0.001)

    epochs = Slider(title="Nb of epochs", value=500, start=1, end=1500, step=50)

    text_output = Paragraph(text="", width=200, height=20)

    train = Button(label="Train model", button_type="success")

    mdn_box = widgetbox(
        mdn_title,
        mdn_text,
        n_gaussians,
        n_layers,
        n_hidden,
        dropout,
        l2,
        epochs,
        train,
        text_output,
    )

    # Plot of p(y/x)

    fit_title = Div(
        text="<h3>Plot Conditional Distribution P(y/x=X) </h3> - You can plot the real p(y/x) after simulating data. <br /> - You can plot estimated p(y/x) after taining the model."
    )

    at_X = TextInput(title="X", value="1.0")

    fit_sim = Button(label="Plot real p(y/x)", button_type="success")
    fit_sim_mdn = Button(label="Plot estimated p(y/x)", button_type="success")

    fit1 = widgetbox(fit_title, at_X, fit_sim, fit_sim_mdn)

    ###-----------------------------------------------------------------------###
    ###-----------------------BASE-LEVEL FUNCTIONS----------------------------###

    def make_data():
        # Sample points
        bi_gmm = SimulateBiGMM(
            int(n_kernels.value),
            float(means_loc.value),
            float(means_scale.value),
            float(covariance_x_loc.value),
            float(covariance_x_scale.value),
            float(covariance_y_loc.value),
            float(covariance_y_scale.value),
        )
        (x, y) = bi_gmm.simulate_xy(int(ctl_nsamp.value))

        return x, y, bi_gmm

    ###-----------------------------------------------------------------------###
    ###------------------DATA SOURCES AND INITIALIZATION----------------------###

    source1 = ColumnDataSource(data=dict(x=[], y=[]))
    plot_data.scatter("x", "y", source=source1, size=3, color="#3A5785", alpha=0.6)

    source2 = ColumnDataSource(data=dict(x=[], prob=[]))
    source3 = ColumnDataSource(data=dict(x=[], prob=[]))

    plot_conditional.line(
        "x", "prob", source=source2, color="#21b093", alpha=0.8, legend="True CDF"
    )

    plot_conditional.line(
        "x", "prob", source=source3, color="orange", alpha=0.8, legend="Estimated CDF"
    )

    def click_simulate():
        # Make it global to be used later
        global x_points, y_points
        global bi_gmm_
        # reset pdf
        source2.data = dict(x=[], prob=[])
        source3.data = dict(x=[], prob=[])

        x_points, y_points, bi_gmm_ = make_data()

        source1.data = dict(x=x_points, y=y_points)

        plot_data.y_range.start = y_points.min() - 1
        plot_data.y_range.end = y_points.max() + 1

        plot_data.x_range.start = x_points.min() - 1
        plot_data.x_range.end = x_points.max() + 1

        plot_data.xaxis.axis_label = "x"
        plot_data.yaxis.axis_label = "y"

        # 3d surface plot

        xx, yy, zz = bi_gmm_.compute_pdf(
            x_points.min(), x_points.max(), y_points.min(), y_points.max()
        )
        source.data = dict(x=xx, y=yy, z=zz)

    plot_sim.on_click(click_simulate)

    # Real P(y/x)
    def plot_real_conditional():

        source3.data = dict(x=[], prob=[])

        ys = np.linspace(y_points.min() - 1, y_points.max() + 1, 1000)
        prob = bi_gmm_.compute_cdf(float(at_X.value), ys)

        source2.data = dict(x=ys, prob=prob)

    fit_sim.on_click(plot_real_conditional)

    # Fit MDN

    def fit_mdn():
        global mdn_model

        mdn_model = MDN_network(
            int(n_hidden.value),
            int(n_layers.value),
            int(n_gaussians.value),
            float(dropout.value),
        )

        optimizer = torch.optim.Adam(
            mdn_model.parameters(), weight_decay=float(l2.value)
        )

        x_tensor = torch.from_numpy(np.float32(x_points))
        y_tensor = torch.from_numpy(np.float32(y_points))

        for e in range(int(epochs.value)):
            l = train_epoch(mdn_model, (x_tensor, y_tensor), optimizer)

        text_output.text = "Taining is done."

    code = "cds.text = 'Training the model... Please wait.';"
    callback = CustomJS(args={"cds": text_output}, code=code)

    train.js_on_click(callback)
    train.on_click(fit_mdn)

    def plot_estimated_conditional():

        ys = np.linspace(y_points.min() - 1, y_points.max() + 1, 1000)

        prob = simulate_condprob_trained(float(at_X.value), mdn_model, ys)

        source3.data = dict(x=ys, prob=prob)

    fit_sim_mdn.on_click(plot_estimated_conditional)

    # Behavior when the "Clear" button is clicked
    def clear_plot():
        source1.data = dict(x=[], y=[])
        source2.data = dict(x=[], prob=[])
        source3.data = dict(x=[], prob=[])
        source.data = dict(x=np.zeros([1, 1]), y=np.zeros([1, 1]), z=np.zeros([1, 1]))
        text_output.text = ""

    plot_clear.on_click(clear_plot)

    ###-----------------------------------------------------------------------###
    ###----------------------------PAGE LAYOUT--------------------------------###

    col_inputs = column(simulate)
    col_output = column(mdn_box, fit1, plot_clear)

    # col_plots = column(plot_data, s3)
    plot_3d = column(surface_title, surface)

    col_plots = layout([[plot_data, plot_3d], [plot_conditional]])

    row_page = row(col_inputs, col_plots, col_output, width=1200)

    # Make a tab with the layout
    tab = Panel(child=row_page, title="Neual Network - Conditional Density Estimation")
    return tab
