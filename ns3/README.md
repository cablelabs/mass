# Using MASS from NS3
We provide an integration with NS3 that allows replay
of TCP and UDP traffic according to the rate time
series produced by the MASS GAN and contextual
parameters such as Wi-Fi and LTE signal strength
automatically induced from the simulation network.
Application contexts are also suported allowing
simulation of different application type mixes
and transition probabilities.

Rate replay is possible in any network where two
nodes are mutually accessible via IPv4 so they can
exchange UDP and TCP packets in both directions.
IPv6 may also be supported in the future.

Example networks are provided to demonstrate use
with a basic CSMA network, as well as more involved
Wi-Fi and LTE networks.

MASS is agnostic to the network used but automatically
detects whether a MASS node has LTE or Wi-Fi capabilities
to determine the signal strength context.

## Setup and Use
An NS3 3.29 docker environment is provided to run the simulation
demonstrations. To build and run the network execute:
```bash
$ ./build.sh  # only needs to be run once
$ ./run.sh <network>
```
where network can be left out for the basic CSMA simulation or
can be `wifi` or `lte` for the Wi-Fi and LTE simulation demos
respectively.

TCP or UDP simulations are determined with an environment variable
in these demonstrators and can be passed in the `PROTO` environment
variable. By default udp is used. To use `TCP` run:
```
PROTO=tcp ./run.sh
```
The default length of the replay is 100 steps, but it can also be changed
with an environment variable `SEQ_LEN`:
```
SEQ_LEN=10 ./run.sh
```

Other simulation parmeters are set in the example network setup
files [mass\_basic\_example.cc](mass_basic_example.cc),
[mass\_wifi\_example.cc](mass_wifi_example.cc),
and [mass\_lte\_example.cc](mass_lte_example.cc).

To integrate mass in your own NS3 environment the [mass.cc](mass.cc)
and [mass.h](mass.h) files need to be made available in your scratch project
space (the simplest way is to just colocate them with your simulation code).

The rate time series are pre-generated for each supported context and the resulting
trace files need to be accesible from your simulation files as well.
A sample [gen\_context.sh](gen_context.sh) script is provided that does this. It
assumes that the MASS REST API is available on localhost but the host may be changed
with the environment variable `MASS_HOST`.

## API
To make use of the MASS API first include the interfaces with:
```c++
#include "mass.h"
```

Then create your `NodeContainer`, `NetDeviceContainer` and `Ipv4InterfaceContainer`
objects as in the [mass\_basic\_example.cc](mass_basic_example.cc) code. The devices
will be set up differently depending on which type of Network/MAC is set up but the
interface containers are setup in the same way, e.g. with:
```c++
    InternetStackHelper internet;
    internet.Install (nodeContainer);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaceContainer = ipv4.Assign (devices);
``` 
Now, assuming we call our `NodeContainer` `nodeContainer` and our
`Ipv4InterfaceContainer` `inerfaceContainer`, we can set up the MASS
replay as follows:
```c++
double runTime = MassHelper(interfaceContainer, nodeContainer).Init();
```
The returned `runTime` variable should then be used as follows to start
and stop the simulation:
```c++
    Simulator::Stop (Seconds (runTime));
    Simulator::Run ();
    Simulator::Destroy ();
```

### Custom trace and client-server nodes 
The above assumes you want to run the replay between the first two nodes
in your node container and also assumes the trace file prefix is `mass.trace`.
To change this behavior to let's say read data from traces generated
in a `/data` directory with the prefix `sample.trace`, and setting up a MASS
replay between node index 3 and 4 in the node container you can run the following:
```c++
    double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetTrace("/data/sample.trace")
      .SetClientIndex(3)
      .SetServerIndex(4)
      .Init();
```
Which nodes you designate as client and server determines how the replay is directed
as the download rates are replayed in messages sent from the server to the client
node and vice versa for uploads.

### Enable context replay
Signal context replay is enabled whenever a network supports Wi-Fi or
LTE endpoints (Uses `WiFiPhy` or `LteUePhy`). Application context
however needs to be enabled explicitly. Application contexts are driven
by transition probabilities that should also be set or otherwise default
to making each application type as probable as any other and continuing
with the same application in the next time step as likely to
continuing to use the same application.
Two application types are supported `STREAM` denoting high throughput
video and audio streaming applications (see MASS GAN documentation),
and `INTERACT` which represents all other applications.
Application contexts can be enabled as follows:
```c++
    double runTime = MassHelper(interfaceContainer, nodeContainer)
      .EnableAppContext()
      .SetInitialApp("INTERACT")
      .SetStreamStayProbability(0.8)
      .SetInteractStayProbability(0.6)
      .Init();
```
This will ensure that an `INTERACT` trace is is replayed in the beginning
and then with `40%` likelihood it will switch over to a `STREAM` trace.
Once in a `STREAM` replay it will switch back to an `INTERACT` replay
with `20%` likelihood. The probabilities are evalated for each time step
also known as epoch.

### Custom Max Upload and Download rate
By default the max download rate is `2Mbps` and the max
download rate is `1Mbps` to change these defaults do the following:
```c++
  double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetMaxUpRate(5)
      .SetMaxDownRate(100)   
      .Init();
```
which sets the maximum upload rate to `5Mbps` and the maximum
download rate to `100Mbps`.

### Custom Message size
During replay both with TCP and UDP messages with packet
size `1024 bits` are replayed. To change the message
size do the following:
```c++
  double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetMessageSize(2048)
      .Init();
```
to change the size of packets to `2048 bits`.

### Setting UDP or TCP traffic
UDP is default but TCP can be set as follows:
```c++
  double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetProtocol("tcp")
      .Init();
```
### Custom epoch length
Each rate in the trace time series will dy default be replayed
for `2 seconds`. This can be changed with:
```c++
  double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetEpochTime(10.5)
      .Init();
```
where each time step is replayed for `10.5 seconds`. 

### Combining customizations
All customizations may be combined in arbitrariy orders
and constellations. E.g all the custumzations above
could be combined into:
```c++
  double runTime = MassHelper(interfaceContainer, nodeContainer)
      .SetEpochTime(10.5)
      .SetProtocol("tcp")
      .SetMessageSize(2048)
      .SetMaxUpRate(5)
      .SetMaxDownRate(100)   
      .EnableAppContext()
      .SetInitialApp("INTERACT")
      .SetStreamStayProbability(0.8)
      .SetInteractStayProbability(0.6)
      .SetTrace("/data/sample.trace")
      .SetClientIndex(3)
      .SetServerIndex(4)
      .Init();
```
Each client and server node pairing may also be configured
independently to, for example, set up one udp and
and one tcp replay concurrently on two pairings you could do:
```c++
  double runTime1 = MassHelper(interfaceContainer, nodeContainer)
      .SetTrace("/data/sample.trace.1")
      .SetClientIndex(0)
      .SetServerIndex(1)
      .SetProtocol("udp")
      .Init();
  double runTime2 = MassHelper(interfaceContainer, nodeContainer)
      .SetTrace("/data/sample.trace.2")
      .SetClientIndex(2)
      .SetServerIndex(3)
      .SetProtocol("tcp")
      .Init();

 Simulator::Stop (Seconds (max(runTime1,runTime2)));
 Simulator::Run ();
 Simulator::Destroy ();
```
Since the endpoints use well-known ports for tcp and udp, a node may only
be an endpoint for a single MASS pairing at a time. In theory a single
node could participate in both an UDP and TCP exchange at the same time,
however this is currently not supported.
