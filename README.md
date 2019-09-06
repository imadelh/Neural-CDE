# Interactive Conditional Density Estimation with Neual Network

This project creates a dashboard to simulate observations (x,y) from a joint distribution p(x,y) and train neural networks to estimate the conditional density p(y/x) (with access to hyper-parameters and training live). 

Live version: https://nn-cde.imadelhanafi.com 

Blog post (for math details): https://imadelhanafi.com/posts/conditional_density_estimation_nn/


**Important**: It is advised to run the code localy and don't rely on the live version as it is just for illustration purpose only. There may be some performance issues when multiple users are training models on the same time since the service is hosted on just 1 machine (no $$ for a cluster with a load balancer). 

<p align="center">
  <img src="https://imadelhanafi.com/img/cde.png" title="figure">
</p>


# Running on your laptop/server

A docker image to run the whole project provided. 

Clone the repo and launch the docker container

```
sudo docker run -it --rm -v ~/bokeh:/app -p 8888:8888 imadelh/pytorch_bokeh_server:v1 bash
```

Then you can run the Bokeh server using one the following commands and you can access the app on localhost:8888.

```
# For testing/dev 

python3 /app/main.py

# To run Bokeh Serve with multiple processes

bokeh serve --show bokeh_server.py --port 8888 --num-procs 4 

```


# Code organization 


The deployed version contains 3 tabs. Code for each tab is provided in a separate independant file

- Tab1: Maximum Likelihood Estimation: `scripts/mle.py`

- Tab2: Kernel Density Estimation: `scripts/kde.py`

- Tab3: Conditional Density Estimation with Neural Networks (MDN): `scripts/nn_cde.py`


Additionally, **Mixture density netwroks** `scripts/mdn.nn` can be used indepently from the project for any kind of data (small changes on the input shape of the network are needed).


[More documentation is needed, but the code follows the logic explained in the blog post]


# Contributions 

Contributions are welcome. If you have ideas about adding new features or support other functionalities, please leave a comment or open a PR. If you wish to host the project on your own servers with more options and computing power (more data, possibility to upload user’s data and train CDE on it, variety of neural networks) please don’t hesitate to contact me and we can set up the service together. 


---

Imad El Hanafi
