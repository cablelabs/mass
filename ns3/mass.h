#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/applications-module.h>
#include <ns3/internet-module.h>
#include <ns3/address.h>

using namespace ns3;

class MassClient : public Application
{
public:
  static const std::vector<std::string> CONTEXTS;

  MassClient ();
  void SetPacketSize(int packetSize);
  void SetRemote (std::string socketType,
                  Address remote);
  void SetRateTraces (std::map<std::string,std::vector<int>> traces);
  void SetEpochTime(Time epoch);
  void SetDirection(std::string direction);
  void SetWiFiSignal(double rssi);
  void SetLTESignal(double rsrq);
  void SetAppContextEnabled(bool enabled);
  void SetInitialApp(std::string app);
  void SetStreamStayProbability(double prob);
  void SetInteractStayProbability(double prob);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);
  void DoGenerate (void);
  std::string GetSignalContext();
  std::string GetContext();
  std::vector<int> GetContextTrace();
  std::string NextApp();

  Time m_start;
  bool m_stopped;
  int m_packetSize;
  bool m_appEnabled;
  Time m_epoch;
  long m_currentEpoch;
  std::string m_direction;
  std::string m_socketType;
  std::map<std::string,std::vector<int>> m_traces;
  double m_wifiSignal;
  double m_lteSignal;
  Ptr<Socket> m_socket;  
  std::string m_currentApp;
  double m_streamStayProbability;
  double m_interactStayProbability;
  std::string m_lastContext;
};

class MassClientAppHelper
{
public:
  MassClientAppHelper (std::string protocol, Address remote);
  void SetPacketSize(int packetSize);
  void SetRateTraces (std::map<std::string,std::vector<int>> traces, std::string direction);
  void SetEpochTime(Time epoch);
  void LoadRateTraces(std::string filename, std::string direction, int maxRate);
  void SetAppContextEnabled(bool enabled);
  void SetInitialApp(std::string app);
  void SetStreamStayProbability(double prob);
  void SetInteractStayProbability(double prob);
  int GetSeqLen();

  ApplicationContainer Install (NodeContainer nodes);
private:
  std::string m_protocol;
  Address m_remote;
  int m_packetSize;
  std::map<std::string,std::vector<int>> m_traces;
  std::string m_direction;
  Time m_epoch;
  double m_streamStayProbability;
  double m_interactStayProbability;
  bool m_appEnabled;
  std::string m_initialApp;


};  


class MassHelper
{
public:
  MassHelper(Ipv4InterfaceContainer interfaceContainer, NodeContainer nodeContainer);
  MassHelper& SetTrace(std::string trace);
  MassHelper& SetMaxUpRate(int maxUpRate);
  MassHelper& SetMaxDownRate(int maxDownRate);
  MassHelper& SetEpochTime(double epochTime);
  MassHelper& SetProtocol(std::string protocol);
  MassHelper& SetMessageSize(int messageSize);
  MassHelper& SetClientIndex(int clientIndex);
  MassHelper& SetServerIndex(int serverIndex);
  MassHelper& EnableAppContext();
  MassHelper& SetInitialApp(std::string app);
  MassHelper& SetStreamStayProbability(double prob);
  MassHelper& SetInteractStayProbability(double prob);

  double Init();
private:
  Ipv4InterfaceContainer m_interfaceContainer;
  NodeContainer m_nodeContainer;
  std::string m_trace;
  int m_maxUpRate;
  int m_maxDownRate;
  double m_epochTime;
  std::string m_protocol;
  int m_messageSize;
  int m_clientIndex;
  int m_serverIndex;
  double m_streamStayProbability;
  double m_interactStayProbability;
  bool m_appEnabled;
  std::string m_initialApp;
};


