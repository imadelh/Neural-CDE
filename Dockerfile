FROM imadelh/pytorch_bokeh_server:v1

# set working directory
ENV HOME /app
WORKDIR $HOME
ADD . $HOME

EXPOSE 8888

CMD bokeh serve --show bokeh_server.py --port 8888 --num-procs 2 --allow-websocket-origin=*
