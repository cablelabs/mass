# Mobile Autonomous Station Simulator (MASS)

This is a companion implementation of the tool described in [MASS: Mobile Autonomous Station Simulation](https://arxiv.org/abs/2111.09161).

When validating network infrastructure innovations in simulations and testbeds, it is common to use one of two approaches:

1. Replay traces from pre-recorded telemetry of real usage
1. Apply distributional models of IAT, load etc

The first approach suffers from:
* privacy liabilities, 
* large amounts of data needed to simulate many users,  
* number of users that can be simulated are capped by measurements, 
* discrepancy between measurement and test environment leading to unrealistic replay load,
* Trace replay does not react to environment

It however tends to provide more accurate, and user(reality)-based load dynamics compared to the second approach of model-based trace generation that is just based on distributional properties. Model-based trace generation also suffers from the lack of responsiveness to environment changes.

So, how can we combine the best of both worlds and achieve:
* no privacy exposure
* realistic trace replay
* limited data needed for deployment
* dynamic to scale to any number of distinct users
* reactive to environment

The solution should be easy to deploy in simulators such as NS3 as well as on real hardware such as routers, and mobile phones as a front-end to tools like iPerf.


Our approach is to develop an autonomous agent that can both replay realistic workloads as well as react to conditions appearing in the environment.

For the first part we propose a GAN-based [1] RNN-LSTM time series generator, and for the latter a reinforcement learning model.

The GAN Generator and Discriminator competitive game training will be done on a real non-anonymized trace and then the Generator RNN model can be shipped to any simulation or experiment environment.
To achieve even greater privacy protection the GAN Generator could be a combination (e.g. with secure aggregation) of many individually trained GAN models. But since even a single GAN generator combines dynamics from many users and can generate an unlimited number of sample traces, such aggregation may not be necessary. Nevertheless, the GAN Generator model should be exportable and easy to enhance locally in deployment, and entities with data should be able to easily generate such models in a standard exportable and sharable format.

Similar time series GANs have been proposed for music score generation in [2]. A survey of spatio-temporal GAN research is available in [3].

In terms of contectual awareness and reaction to changing conditions we are mainly interested in task and load awareness.
Task awareness is which app is running, as it would impact the network demand. Similarly load awareness is the signal strength,
RSSI or similar mesures capping the achievable throughput. One strawman approach is to generate different time series of
network usage based on different conditions, i.e. apps, RSSIs. Then apply some transition probabilities to
"jump" between the parallell timelines. In an experiment these jumps could also be triggered by the experiment conditions,
e.g. when a station moves closer to an AP or base station it jumps to the "high-signal" timeline.


[1] Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

[2] Mogren, Olof. "C-RNN-GAN: Continuous recurrent neural networks with adversarial training." arXiv preprint arXiv:1611.09904 (2016).

[3] Gao, Nan, Hao Xue, Wei Shao, Sichen Zhao, Kyle Kai Qin, Arian Prabowo, Mohammad Saiedur Rahaman, and Flora D. Salim. "Generative adversarial networks for spatio-temporal data: A survey." arXiv preprint arXiv:2008.08903 (2020).

## Getting Started
The following setup has been tested on Ubuntu 22.04.1, but should work on any Linux/Mac platform.

## Dependencies
```
$ sudo apt-get install python3 python3-pip screen r-base curl unzip iperf3
$ pip3 install sh flask numpy scipy
$ pip3 install torch --no-cache-dir 
$ mkdir -p models
$ curl https://raw.githubusercontent.com/cjbayron/c-rnn-gan.pytorch/master/c_rnn_gan.py > models/c_rnn_gan.py
```

## Data
The sample dataset (mpud.zip) used can be downloaded from [Telefonica Mobile Phone Use Dataset](https://sites.google.com/view/mobile-phone-use-dataset).
First extract the archive, and put all csv files in the data subdirectory 
```
$ mkdir -p data
$ unzip mpud.zip
$ cp mobile_phone_use/data/*.csv data/
```

and then run:
```
$ ./bin/tometrics.sh
$ ./bin/tomass.sh
```
## Train
Run: 
```
$ ./bin/context.sh 
$ tail -f screenlog.0
```
Note, training takes a long time on a CPU, so a GPU (CUDA/NVidia) is recommended if available.

## Contexts
Training is done based on contextual parameters that are determined based on
impact on results. The full list of contextual parameters are:
```
LOW (low Wi-Fi signal, < -75 RSSI)
HIGH (high Wi-Fi signal, >= -75 RSSI)
STREAM (streaming app: MUSIC_AND_AUDIO, MAPS_AND_NAVIGATION, SPORTS, VIDEO_PLAYERS)
INTERACT (non-streaming app)
STREAM_LOW (STREAM and LOW)
STREAM_HIGH (STREAM and HIGH)
INTERACT_LOW (INTERACT and LOW)
INTERACT_HIGH (INTERACT and HIGH)
```
Not all contexts will be trained with a separate GAN generator. Before
training, the data will be examined to see which contexts show a significant
difference in upload or download volume (>10%). Only those contexts will then
be trained and get a dedicated GAN generator. If a context is specified that
was determined not to be significant it will fall back on a DEFAULT context
which is the non context-aware GAN generator trained on all data.

The significant contexts can be obtained from:
```
cat data/contexts
```
## Generate CLI
To generate trace data from the CLI run
```
$ CONTEXT=$CONTEXT SAMPLES=$SAMPLES SEQ_LEN=$SEQ_LEN ./bin/gengan.sh
```
this will result in a trace being written to the directory
```
  ./gan${CONTEXT}/
```

## Generate Python API
The generator can be accessed through python as follows (e.g. from `PYTHONPATH=./scripts/ python3`):
```
from mass import Mass
users = 100
seq_len = 12
normalize = "minmax" # or "pos" for x-min vs (x-min)/(max-min)
context = "LOW" # or any other context from the list above
do_shuffle = False # whether to time shift traces across users
                   # randomly
m = Mass(users,seq_len,do_shuffle=do_shuffle,normalize=normalize)
data = m.generate(context=context)
```
## Generate REST API
A flask server can be started with:
```
python3 scripts/mass_server.py
```
It can then be accessed with, e.g.:
```
curl -d '{"context":"LOW","seq_len":12,"users":10,"normalize":"minmax","shuffle":false}' http://localhost:7777/generate
```
## MASS Client
A mass bash client depending only on `curl` and `iperf` has been developed
to run trace replays.

It interacts with the MASS server and iperf (default) or our custom performance
servers.

To test it start the REST API server and iperf
```
$ python3 scripts/mass_server.py
...
$ iperf3 -s
```
then run the client with
```
./bin/mass
```

To configure the client, create a config called custom.config with
at least the following content:
```
MASS_HOST=masshost
PERF_HOST=perfhost
```
and then run
```
./bin/mass custom.config
```
Or you can run:
```
MASS_HOST=masshost PERF_HOST=perfhost ./bin/mass
```

By default only downloads (iperf server to iperf client) are performed and only tcp
traffic, assuming an iperf3 server is running on the `PERF_HOST` (port 5201).

Study the [default config](bin/default.config) for documentation
on what settings can be customized.

## NS3 Client
For details on how to use MASS with NS3 look at the separate [NS3 tutorial](ns3/README.md).
