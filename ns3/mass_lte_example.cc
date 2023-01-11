#include <fstream>
#include <iostream>
#include <sstream>

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/mobility-helper.h>
#include <ns3/applications-module.h>
#include <ns3/internet-module.h>
#include <ns3/mobility-model.h>
#include <ns3/lte-module.h>
#include <ns3/config-store.h>
#include <ns3/traffic-control-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/rectangle.h>

#include "mass.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MassLTEExample");


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

  // LTE setup
  int numBS = 2;
  int gridSize = 3500;

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<LteHexGridEnbTopologyHelper> lteHexGridEnbTopologyHelper = CreateObject<LteHexGridEnbTopologyHelper> ();
  lteHexGridEnbTopologyHelper->SetLteHelper(lteHelper);

  int sitesPerRow = sqrt(numBS)/3;
  int siteHeight = 3*(gridSize)/sqrt(numBS);
  int interSiteDistance = siteHeight;
  int sectorOffset = interSiteDistance / 3;

  lteHexGridEnbTopologyHelper->SetAttribute ("InterSiteDistance", DoubleValue (interSiteDistance));
  lteHexGridEnbTopologyHelper->SetAttribute ("GridWidth", UintegerValue (sitesPerRow));
  lteHexGridEnbTopologyHelper->SetAttribute ("SectorOffset", DoubleValue (sectorOffset));
  lteHexGridEnbTopologyHelper->SetAttribute ("MinX", DoubleValue (-gridSize+interSiteDistance*1.5));
  lteHexGridEnbTopologyHelper->SetAttribute ("MinY", DoubleValue (-gridSize+interSiteDistance/4));
  lteHexGridEnbTopologyHelper->SetAttribute ("SiteHeight", DoubleValue (siteHeight));


  lteHelper->SetEnbAntennaModelType ("ns3::ParabolicAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Beamwidth",   DoubleValue (70));
  lteHelper->SetEnbAntennaModelAttribute ("MaxAttenuation",     DoubleValue (20.0));

  NodeContainer enbNodes;
  enbNodes.Create (numBS);


  // mobility setup
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel"),
  mobility.Install (nodeContainer.Get(1));

  std::string format = "ns3::UniformRandomVariable[Min="; 
  format += std::to_string(-gridSize);
  format += "|Max=";
  format += std::to_string(gridSize);
  format += "]";
  mobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                  "X", StringValue (format),
                                  "Y", StringValue (format));
  mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
  "Bounds", RectangleValue (Rectangle (-gridSize, gridSize, -gridSize, gridSize)));
  mobility.Install (nodeContainer.Get(0));


  NetDeviceContainer enbDevs;
  enbDevs = lteHexGridEnbTopologyHelper->SetPositionAndInstallEnbDevice (enbNodes);

  NetDeviceContainer devices;
  devices = lteHelper->InstallUeDevice (nodeContainer);

  lteHelper->AttachToClosestEnb (devices, enbDevs);

  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VIDEO;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (devices, bearer);

  
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


  NS_LOG_INFO ("MassLTEExample: Running Simulation");
  Simulator::Stop (Seconds (runTime));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("MassLTEExample: Done Running Simulation");

  return;
}

int main (int argc, char *argv[])
{
  LogComponentEnable ("Mass", LOG_LEVEL_INFO);
  LogComponentEnable ("MassLTEExample", LOG_LEVEL_INFO);
  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults ();
  run_sim();
  return 0;
}

