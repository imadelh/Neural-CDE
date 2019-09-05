"""

To run with bokeh server

"""

from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

from scripts.mle import mle
from scripts.kde import kde
from scripts.nn_cde import nn


tab1 = mle()
tab2 = kde()
tab3 = nn()


# Put all the tabs into one application
tabs = Tabs(tabs=[tab1, tab2, tab3])

# Put the tabs in the current document for display
curdoc().add_root(tabs)
curdoc().title = "Conditional Density Estimation - by Imad El"
