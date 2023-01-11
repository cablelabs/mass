#include <fstream>
#include <iostream>
#include <sstream>

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/csma-module.h>
#include <ns3/applications-module.h>
#include <ns3/internet-module.h>
#include "ns3/traffic-control-module.h"
#include "ns3/point-to-point-module.h"

#include "mass.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MassBasicExample");


void run_sim()
{

  NodeContainer nodeContainer;
  nodeContainer.Create (2);

  // MAC setup
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1400));
  NetDeviceContainer d = csma.Install (nodeContainer);
 
  // Internet setup
  InternetStackHelper internet;
  internet.Install (nodeContainer);
 
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaceContainer = ipv4.Assign (d);

  // Mass setup
  double runTime = MassHelper(interfaceContainer, nodeContainer)
                      .SetTrace("/sim/data/sample.trace")
                      .SetProtocol(std::getenv("PROTO"))
                      .Init();

  NS_LOG_INFO ("MassBasicExample: Running Simulation");
  Simulator::Stop (Seconds (runTime));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("MassBasicExample: Done Running Simulation");

  return;
}

int main (int argc, char *argv[])
{
  LogComponentEnable ("Mass", LOG_LEVEL_INFO);
  LogComponentEnable ("MassBasicExample", LOG_LEVEL_INFO);
  run_sim();
  return 0;
}

