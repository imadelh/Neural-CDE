import numpy as np

from scipy.stats import norm, gamma, beta
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Panel
from bokeh.layouts import column, row, widgetbox
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


DISTRIBUTIONS = {"Gaussian": norm, "Beta": beta, "Gamma": gamma}

KERNELS = ["gaussian", "tophat", "exponential", "linear", "cosine"]


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


def kde():
    ###------------------------PARAMETER DEFAULTS-----------------------------###
    #  range[Lower, Upper, Step Size]
    ### - SAMPLING Parameters
    d_nsamp, r_nsamp = 500, [100, 2000, 50]  # Number of samples

    plot_data = figure(
        plot_height=400,
        plot_width=800,
        title="Data Histogram",
        toolbar_location="above",
        x_axis_label="x",
        y_axis_label="Density",
        tools="pan,save,box_zoom,wheel_zoom",
    )
    style(plot_data)

    plot_clear = Button(label="Clear All", button_type="warning")

    # Plot Control Buttons

    ctl_title = Div(text="<h3>Simulator</h3>")
    dist_type = Select(
        title="Select sampling distribution:",
        value="Gaussian",
        options=["Gaussian", "Beta", "Gamma"],
    )

    div1 = Div(
        text="""<p style="border:3px; border-style:solid; border-color:grey; padding: 1em;">
                    Parameters depend on the distribution. Refer to Scipy Documentation. <br />  
                    - Gaussian: loc (mean), scale (variance).<br />  
                    - Gamma: a, loc, scale.<br /> 
                    - Beta: a, b, loc, scale.<br /> 
                    </p>""",
        width=300,
        height=130,
    )

    ctl_nsamp = Slider(
        title="Number of samples",
        value=d_nsamp,
        start=r_nsamp[0],
        end=r_nsamp[1],
        step=r_nsamp[2],
    )

    mu = TextInput(title="Mean", value="0.0")
    sigma = TextInput(title="Variance", value="1.0")
    a = TextInput(title="a", value="1.0")
    b = TextInput(title="b", value="1.0")

    plot_sim = Button(label="Simulate", button_type="primary")
    simulate = widgetbox(
        ctl_title, dist_type, div1, ctl_nsamp, mu, sigma, a, b, plot_sim
    )

    # plot_ctls = column(ctl_title, div1, plot_sim)

    ### Manual Fitting
    fit_title = Div(text="<h3>Manual KDE</h3>")

    kernel = Select(
        title="Select kernel to fit on data:",
        value="gaussian",
        options=["gaussian", "tophat", "exponential", "linear", "cosine"],
    )
    bandwidth = TextInput(title="Bandwidth", value="1.0")

    fit_sim = Button(label="Fit", button_type="success")

    fit1 = widgetbox(fit_title, kernel, bandwidth, fit_sim)

    ### Cross Validation fit
    fit_title = Div(
        text="<h3>Cross Validation - Params Search</h3> <br /> Select kernels for parameter search."
    )

    kernels = CheckboxButtonGroup(
        active=[0, 1], labels=["gaussian", "tophat", "exponential", "linear", "cosine"]
    )
    bandwidths = RangeSlider(
        title="Bandwidth in Log-space", start=-2, end=1, value=(-1, 0.5), step=0.05
    )

    cv_slider = Slider(title="Nb of cross-validation", value=5, start=1, end=10, step=1)

    fit_sim2 = Button(label="Fit", button_type="success")
    text_output = TextAreaInput(
        value="Choose Bandwidth Range and Kernels from above.",
        rows=4,
        title="Estimated Paramters:",
    )

    fit2 = widgetbox(fit_title, kernels, bandwidths, cv_slider, fit_sim2, text_output)

    ###-----------------------------------------------------------------------###
    ###-----------------------BASE-LEVEL FUNCTIONS----------------------------###

    def make_data(dist_type, params):
        # sample data according to dist_type
        data = DISTRIBUTIONS[dist_type].rvs(**params)
        return data

    ###-----------------------------------------------------------------------###
    ###------------------DATA SOURCES AND INITIALIZATION----------------------###

    source1 = ColumnDataSource(data=dict(hist=[], left=[], right=[]))
    plot_data.quad(
        source=source1,
        bottom=0,
        top="hist",
        left="left",
        right="right",
        fill_color="blue",
        line_color="white",
        alpha=0.5,
    )

    source2 = ColumnDataSource(data=dict(x=[], y=[]))
    plot_data.line(
        "x",
        "y",
        source=source2,
        line_width=2,
        alpha=0.7,
        legend="Estimated PDF",
        line_color="black",
    )

    def click_simulate():
        # Make it global to be used later
        global d_data
        # reset pdf
        source2.data = dict(x=[], y=[])
        text_output.value = ""

        if dist_type.value == "Gaussian":
            params = {
                "loc": float(mu.value),
                "scale": float(sigma.value),
                "size": int(ctl_nsamp.value),
            }

        elif dist_type.value == "Gamma":
            params = {
                "loc": float(mu.value),
                "scale": float(sigma.value),
                "a": float(a.value),
                "size": int(ctl_nsamp.value),
            }

        elif dist_type.value == "Beta":
            params = {
                "loc": float(mu.value),
                "scale": float(sigma.value),
                "a": float(a.value),
                "b": float(b.value),
                "size": int(ctl_nsamp.value),
            }

        d_data = make_data(dist_type.value, params)

        hist, edges = np.histogram(d_data, density=True, bins=100)

        source1.data = dict(hist=hist, left=edges[:-1], right=edges[1:])

        plot_data.y_range.start = 0
        plot_data.y_range.end = 1.2 * hist.max()

        plot_data.x_range.start = edges.min() - 1 * (dist_type.value != "Beta")
        plot_data.x_range.end = edges.max() + 1 * (dist_type.value != "Beta")

        plot_data.xaxis.axis_label = "x"
        plot_data.yaxis.axis_label = "Density"

    plot_sim.on_click(click_simulate)

    # KDE Fit

    def fit_kde():
        kde = KernelDensity(kernel=kernel.value, bandwidth=float(bandwidth.value)).fit(
            d_data.reshape(-1, 1)
        )

        x = np.linspace(d_data.min()-0.5, d_data.max()+0.5, 1000).reshape(-1, 1)
        log_dens = kde.score_samples(x)

        source2.data = dict(x=x, y=np.exp(log_dens))

    fit_sim.on_click(fit_kde)

    # Fir CV-model

    def fit_kde_cv():
        text_output.value = "Running..."

        kernels_space = [KERNELS[i] for i in kernels.active]

        min_b = list(bandwidths.value)[0]
        max_b = list(bandwidths.value)[1]

        params = {"bandwidth": np.logspace(min_b, max_b, 10), "kernel": kernels_space}

        grid = GridSearchCV(KernelDensity(), params, cv=int(cv_slider.value), iid=False)

        grid.fit(d_data.reshape(-1, 1))

        x = np.linspace(d_data.min()-0.5, d_data.max()+0.5, 1000).reshape(-1, 1)
        log_dens = grid.best_estimator_.score_samples(x)

        source2.data = dict(x=x, y=np.exp(log_dens))

        text_output.value = "CV done. \nBest Params: \n" + str(grid.best_params_)

    fit_sim2.on_click(fit_kde_cv)

    # Behavior when the "Clear" button is clicked
    def clear_plot():
        source1.data = dict(hist=[], left=[], right=[])
        source2.data = dict(x=[], y=[])
        text_output.value = ""

    plot_clear.on_click(clear_plot)

    ###-----------------------------------------------------------------------###
    ###----------------------------PAGE LAYOUT--------------------------------###

    col_inputs = column(simulate)
    col_output = column(fit1, fit2, plot_clear)
    col_plots = column(plot_data)
    row_page = row(col_inputs, col_plots, col_output, width=1200)

    # Make a tab with the layout
    tab = Panel(child=row_page, title="Kernel Density Estimation")
    return tab
