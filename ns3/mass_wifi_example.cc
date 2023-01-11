#include <fstream>
#include <iostream>
#include <sstream>

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/csma-module.h>
#include <ns3/yans-wifi-helper.h>
#include <ns3/ssid.h>
#include <ns3/mobility-helper.h>
#include <ns3/applications-module.h>
#include <ns3/internet-module.h>
#include <ns3/yans-wifi-channel.h>
#include <ns3/mobility-model.h>
#include <ns3/traffic-control-module.h>
#include <ns3/point-to-point-module.h>

#include "mass.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MassWiFiExample");


void
CourseChange (std::string context, Ptr<const MobilityModel> model)
{
  Vector position = model->GetPosition ();
  NS_LOG_UNCOND (context <<
    " x = " << position.x << ", y = " << position.y);
}

void run_sim()
{

  NodeContainer nodeContainer;
  nodeContainer.Create (2);

  // WiFi setup
  std::string phyMode ("DsssRate1Mbps");
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                       StringValue (phyMode));
  WifiHelper wifi;
  YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default ();
  wifiPhy.Set ("RxGain", DoubleValue (0) );
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiPhy.SetChannel (wifiChannel.Create ());
  WifiMacHelper wifiMac;
  Ssid ssid = Ssid ("wifi-default");
  wifiMac.SetType ("ns3::StaWifiMac",
                    "Ssid", SsidValue (ssid));
  NetDeviceContainer staDevice = wifi.Install (wifiPhy, wifiMac, nodeContainer.Get (0));
  NetDeviceContainer devices = staDevice;
  wifiMac.SetType ("ns3::ApWifiMac",
                    "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevice = wifi.Install (wifiPhy, wifiMac, nodeContainer.Get (1));
  devices.Add (apDevice);

  // mobility setup
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodeContainer.Get(1));
  mobility.SetMobilityModel ("ns3::RandomDirection2dMobilityModel");
  mobility.Install (nodeContainer.Get(0));

  Config::Connect ("/NodeList/0/$ns3::MobilityModel/CourseChange", MakeCallback (&CourseChange));

  // Internet setup
  InternetStackHelper internet;
  internet.Install (nodeContainer);
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaceContainer = ipv4.Assign (devices);

  // Mass setup
  double runTime = MassHelper(interfaceContainer, nodeContainer)
                      .SetTrace("/sim/data/sample.trace")
                      .SetProtocol(std::getenv("PROTO"))
                      .EnableAppContext()
                      .Init();


  NS_LOG_INFO ("MassWiFiExample: Running Simulation");
  Simulator::Stop (Seconds (runTime));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("MassWiFiExample: Done Running Simulation");

  return;
}

int main (int argc, char *argv[])
{
  LogComponentEnable ("Mass", LOG_LEVEL_INFO);
  LogComponentEnable ("MassWiFiExample", LOG_LEVEL_INFO);
  run_sim();
  return 0;
}

