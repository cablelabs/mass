#include "mass.h"

#include <ns3/yans-wifi-helper.h>


NS_LOG_COMPONENT_DEFINE ("Mass");

const std::vector<std::string> MassClient :: CONTEXTS ({ "DEFAULT", "INTERACT", "INTERACT_HIGH", "INTERACT_LOW",
                                "STREAM", "STREAM_LOW", "STREAM_HIGH", "LOW", "HIGH" });

void
WiFiMonitorSniffer (Ptr<MassClient> massClient, Ptr< const Packet > packet, uint16_t channelFreqMhz, WifiTxVector txVector, MpduInfo aMpdu, SignalNoiseDbm signalNoise)
{
  massClient->SetWiFiSignal(signalNoise.signal);
}

void 
LteMonitorSniffer (Ptr<MassClient> massClient, uint16_t rnti, uint16_t cellId, 
                   double rsrp, double rsrq, bool servingCell, uint8_t componentCarrierId)
{
  massClient->SetLTESignal(rsrq);
}	


MassClient::MassClient()
{
}

void 
MassClient::SetPacketSize(int packetSize) 
{
  m_packetSize = packetSize;
}

void 
MassClient::SetWiFiSignal(double rssi) 
{
  m_wifiSignal = rssi;
}

void 
MassClient::SetLTESignal(double rsrq) 
{
  m_lteSignal = rsrq;
}

void
MassClient::SetAppContextEnabled(bool enabled)
{
  m_appEnabled = enabled;
}

void
MassClient::SetInitialApp(std::string app)
{
  m_currentApp = app;
}

void
MassClient::SetStreamStayProbability(double prob)
{
  m_streamStayProbability = prob;
}

void
MassClient::SetInteractStayProbability(double prob)
{
  m_interactStayProbability = prob;
}

void 
MassClient::SetRemote (std::string socketType, 
                            Address remote)
{
  TypeId tid;
  m_socketType = socketType;
  if (socketType.compare("udp") == 0) {
    tid = ns3::UdpSocketFactory::GetTypeId();
  } else {
    tid = ns3::TcpSocketFactory::GetTypeId();
  }
  m_socket = Socket::CreateSocket (GetNode(), tid);
  m_socket->Bind ();
  m_socket->ShutdownRecv ();
  m_socket->Connect (remote);
}

void 
MassClient::SetRateTraces (std::map<std::string,std::vector<int>> traces)
{
  m_traces = traces;
}

void 
MassClient::SetEpochTime(Time epoch) {
  m_epoch = epoch;
}

void 
MassClient::SetDirection(std::string direction) {
  m_direction = direction;
}

std::vector<int>
MassClient::GetContextTrace()
{
  std::string context = GetContext();
  m_lastContext = context;
  if (m_traces.count(context) > 0) {
    return m_traces[context];
  } else {
    return m_traces["DEFAULT"];
  }
}

std::string
MassClient::NextApp()
{
  double r = (rand() % 100)/100.0;
  double stay_prob;
  bool isStream = false;
  if (m_currentApp.compare("STREAM") == 0) {
    stay_prob = m_streamStayProbability;
    isStream = true;
  } else {
    stay_prob = m_interactStayProbability;
  }
  if (stay_prob >= r) {
    if (isStream) {
      m_currentApp = "INTERACT";
    } else {
      m_currentApp = "STREAM";
    }
  }
  return m_currentApp;
}

void
MassClient::DoGenerate (void)
{
  if (m_stopped) {
    return;
  }
  std::vector<int> trace =  GetContextTrace();
  long currentEpoch =
      std::lround((Simulator::Now().GetSeconds()-m_start.GetSeconds())  /m_epoch.GetSeconds());
  if (currentEpoch < (unsigned)m_traces["DEFAULT"].size()) {
    int rate = trace[currentEpoch];
    if (rate < m_packetSize) {
      rate = m_packetSize;
    }
    double delay = (double)m_packetSize/(double)rate;
    Simulator::Schedule (Seconds (delay), 
                &MassClient::DoGenerate, this);
    Ptr<Packet> p = Create<Packet> (m_packetSize);
    if (currentEpoch != m_currentEpoch) {
    NS_LOG_INFO ("MassClient: " <<  m_socketType  << " " << m_direction << " " 
                 << Simulator::Now().GetSeconds() << " Epoch " << currentEpoch 
                 << " rate " << rate << " delay " << delay << " wifi signal " << m_wifiSignal
                 << " lte signal " << m_lteSignal << " context " << m_lastContext);
      m_currentEpoch = currentEpoch;
    }
    m_socket->Send (p);
  }
}

void 
MassClient::StartApplication (void) {
  m_start = Simulator::Now();
  m_stopped = false;
  DoGenerate();
}

void 
MassClient::StopApplication (void) {
  m_stopped = true;
}

std::string
MassClient::GetSignalContext (void) {
  if (m_wifiSignal == 0 && m_lteSignal == 0) {
    return "DEFAULT";
  }
  if ((m_wifiSignal > -75 && m_wifiSignal != 0) || 
      (m_lteSignal > -15 && m_lteSignal != 0)) {
    return "HIGH";
  }
  return "LOW";
}

std::string
MassClient::GetContext (void) {
  std::string signalContext = GetSignalContext();
  if (!m_appEnabled) {
    return signalContext;
  }
  std::string app = NextApp();
  if (signalContext.compare("DEFAULT") == 0) {
    return app;
  }
  return app + "_" + signalContext;
}


MassClientAppHelper::MassClientAppHelper (std::string protocol, 
                            Address remote)
{
  m_protocol = protocol;
  m_remote = remote;
}

void
MassClientAppHelper::SetPacketSize(int packetSize)
{
  m_packetSize = packetSize;
}

void
MassClientAppHelper::SetAppContextEnabled(bool enabled)
{
  m_appEnabled = enabled;
}

void
MassClientAppHelper::SetInitialApp(std::string app)
{
  m_initialApp = app;
}

void
MassClientAppHelper::SetStreamStayProbability(double prob)
{
  m_streamStayProbability = prob;
}

void
MassClientAppHelper::SetInteractStayProbability(double prob)
{
  m_interactStayProbability = prob;
}

int
MassClientAppHelper::GetSeqLen()
{
  return m_traces["DEFAULT"].size();
}


void
MassClientAppHelper::SetRateTraces (std::map<std::string,std::vector<int>> traces, std::string direction)
{
  m_traces = traces;
  m_direction = direction;
}

void 
MassClientAppHelper::SetEpochTime(Time epoch) {
  m_epoch = epoch;
}



ApplicationContainer 
MassClientAppHelper::Install (NodeContainer nodes)
{
  ApplicationContainer applications;
  for (NodeContainer::Iterator i = nodes.Begin (); i != nodes.End (); ++i)
    {
      Ptr<MassClient> app = CreateObject<MassClient> ();
      app->SetPacketSize (m_packetSize);
      app->SetRateTraces (m_traces);
      app->SetEpochTime (m_epoch);
      app->SetDirection (m_direction);
      app->SetAppContextEnabled(m_appEnabled);
      app->SetInitialApp(m_initialApp);
      app->SetStreamStayProbability(m_streamStayProbability);
      app->SetInteractStayProbability(m_interactStayProbability);
      app->SetWiFiSignal(0);
      app->SetLTESignal(0);
      (*i)->AddApplication (app);
      app->SetRemote (m_protocol, m_remote);
      applications.Add (app);
      std::ostringstream oss;
      oss << "/NodeList/"
        <<  app->GetNode()->GetId()
        << "/DeviceList/*/Phy/MonitorSnifferRx";
      Config::ConnectWithoutContext (oss.str(), MakeBoundCallback (&WiFiMonitorSniffer,app));
      oss.str("");
      oss << "/NodeList/"
        <<  app->GetNode()->GetId()
        << "/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements";
      Config::ConnectWithoutContext (oss.str(), MakeBoundCallback (&LteMonitorSniffer,app));
    }
  return applications;
}

void
MassClientAppHelper::LoadRateTraces(std::string fileName, std::string direction, int maxRate)
{
  std::map<std::string,std::vector<int>> traces;
  for (std::string context: MassClient::CONTEXTS) { 
    std::ifstream inFile;
    std::ostringstream contextFileName;
    contextFileName << fileName << "." << context;
    inFile.open(contextFileName.str());
    std::vector<int> trace = std::vector<int>();
    if (!inFile) {
      NS_LOG_ERROR("MassClientAppHelper: Failed to open file " << fileName); 
      continue;
    }
    float down, up;
    while (inFile >> down >> up) {
      if (direction.compare("down") == 0) {
         trace.push_back((int)(down * maxRate * 1e6));
      } else {
         trace.push_back((int)(up * maxRate  * 1e6));
      }
    }
    traces[context] = trace;
  }
  SetRateTraces(traces, direction);
}

double
MassHelper::Init()
{

  bool isUDP = (m_protocol.compare("udp") == 0);
  int seqLen = 0;
  uint16_t port;
  ApplicationContainer transportApps;
  if (isUDP) {
    port = 4000;
    UdpServerHelper udpServer (port);
    transportApps = udpServer.Install (m_nodeContainer);
  } else {
    port = 5000;
    PacketSinkHelper tcpServer ("ns3::TcpSocketFactory",InetSocketAddress (Ipv4Address::GetAny (), port));
    transportApps = tcpServer.Install (m_nodeContainer);
  }
  InetSocketAddress remoteDown = InetSocketAddress (m_interfaceContainer.GetAddress (m_clientIndex), port);
  InetSocketAddress remoteUp = InetSocketAddress (m_interfaceContainer.GetAddress (m_serverIndex), port);

  // UP client-server 
  MassClientAppHelper up (m_protocol,remoteUp);
  up.SetPacketSize(m_messageSize);
  up.SetAppContextEnabled(m_appEnabled);
  up.SetInitialApp(m_initialApp);
  up.SetStreamStayProbability(m_streamStayProbability);
  up.SetInteractStayProbability(m_interactStayProbability);
  up.LoadRateTraces(m_trace,"up", m_maxUpRate);
  up.SetEpochTime(Seconds(m_epochTime));
  ApplicationContainer upApps = up.Install (m_nodeContainer.Get (m_clientIndex));

  seqLen = up.GetSeqLen();

  // Down client-server 
  MassClientAppHelper down (m_protocol,remoteDown);
  down.SetPacketSize(m_messageSize);
  down.SetAppContextEnabled(m_appEnabled);
  down.SetInitialApp(m_initialApp);
  down.SetStreamStayProbability(m_streamStayProbability);
  down.SetInteractStayProbability(m_interactStayProbability);
  down.LoadRateTraces(m_trace,"down", m_maxDownRate);
  down.SetEpochTime(Seconds(m_epochTime));
  ApplicationContainer downApps = down.Install (m_nodeContainer.Get (m_serverIndex));

  // Scheduling
  transportApps.Start (Seconds (1.0));
  transportApps.Stop (Seconds (seqLen*m_epochTime));
  upApps.Start (Seconds (2.0));
  upApps.Stop (Seconds (seqLen*m_epochTime));
  downApps.Start (Seconds (2.0));
  downApps.Stop (Seconds (seqLen*m_epochTime));

  return seqLen * m_epochTime;
}

MassHelper::MassHelper(Ipv4InterfaceContainer interfaceContainer, NodeContainer nodeContainer)
{
    m_interfaceContainer = interfaceContainer;
    m_nodeContainer = nodeContainer;
    m_trace = "mass.trace";
    m_maxUpRate = 1;
    m_maxDownRate = 2;
    m_epochTime = 2;
    m_protocol = "udp";
    m_messageSize = 1024;
    m_clientIndex = 0;
    m_serverIndex = 1;
    m_appEnabled = false;
    m_initialApp = "INTERACT";
    m_streamStayProbability = 0.5;
    m_interactStayProbability = 0.5;
}

MassHelper&
MassHelper::SetTrace(std::string trace)
{
  m_trace = trace;
  return *this;
}

MassHelper&
MassHelper::SetMaxUpRate(int maxUpRate)
{
  m_maxUpRate = maxUpRate;
  return *this;
}

MassHelper&
MassHelper::SetMaxDownRate(int maxDownRate)
{
  m_maxDownRate = maxDownRate;
  return *this;
}

MassHelper&
MassHelper::SetEpochTime(double epochTime)
{
  m_epochTime = epochTime;
  return *this;
}

MassHelper&
MassHelper::SetProtocol(std::string protocol)
{
  m_protocol = protocol;
  return *this;
}

MassHelper&
MassHelper::SetMessageSize(int messageSize)
{
  m_messageSize = messageSize;
  return *this;
}

MassHelper&
MassHelper::SetClientIndex(int clientIndex)
{
  m_clientIndex = clientIndex;
  return *this;
}

MassHelper&
MassHelper::SetServerIndex(int serverIndex)
{
  m_serverIndex = serverIndex;
  return *this;
}

MassHelper& 
MassHelper::EnableAppContext()
{
  m_appEnabled = true;
  return *this;
}

MassHelper& 
MassHelper::SetInitialApp(std::string app)
{
  m_initialApp = app;
  return *this;
}


MassHelper& 
MassHelper::SetStreamStayProbability(double prob) 
{
  m_streamStayProbability = prob;
  return *this;
}

MassHelper&
MassHelper::SetInteractStayProbability(double prob)
{
  m_interactStayProbability = prob;
  return *this;
}


