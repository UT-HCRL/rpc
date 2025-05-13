#pragma once

#include "configuration.hpp"
#include "controller/interrupt_handler.hpp"

class G1ControlArchitecture;
class G1ControlArchitecture_WBIC;

class G1InterruptHandler : public InterruptHandler {
public:
  G1InterruptHandler(G1ControlArchitecture_WBIC *ctrl_arch);
  G1InterruptHandler(G1ControlArchitecture *ctrl_arch);
  virtual ~G1InterruptHandler() = default;

  void Process() override;

private:
  G1ControlArchitecture_WBIC *ctrl_arch_wbic_;
  G1ControlArchitecture *ctrl_arch_;
};
