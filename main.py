"""

to run without bokeh server.
python3 main.py

"""
from bokeh.server.server import Server
from bokeh.models.widgets import Tabs

from scripts.mle import mle
from scripts.kde import kde
from scripts.nn_cde import nn


def app(doc):
    # Create each of the tabs
    tab1 = mle()
    tab2 = kde()
    tab3 = nn()

    # Put all the tabs into one application
    tabs = Tabs(tabs=[tab1, tab2, tab3])

    # Put the tabs in the current document for display
    doc.add_root(tabs)
    doc.title = "Conditional Density Estimation - by Imad El"


server = Server({"/": app}, num_procs=2, port=8888)
server.start()

if __name__ == "__main__":
    print("Opening Bokeh application on http://localhost:port/")

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
