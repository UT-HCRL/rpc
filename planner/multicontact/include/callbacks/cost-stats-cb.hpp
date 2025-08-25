#include "crocoddyl/core/utils/callbacks.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

class CostRecorderCallback : public crocoddyl::CallbackAbstract {
public:
    CostRecorderCallback(
        const std::vector<std::shared_ptr<crocoddyl::CostModelSum>>& running_cost_models,
        std::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model,
        std::function<double(const std::string&, int)> fetch_value,
        const std::string& stats_csv = "ddp_stats.csv",
        const std::string& terms_csv = "ddp_terms.csv",
        bool include_terminal = false)
        : running_cost_models_(running_cost_models),
          terminal_cost_model_(std::move(terminal_cost_model)),
          fetch_value_(std::move(fetch_value)),
          include_terminal_(include_terminal),
          stats_csv_(stats_csv),
          terms_csv_(terms_csv)
    {
        stats_buffer_ << "iter,cost,step,stop,dV_exp,dV\n";

        std::set<std::string> names_set;
        node_has_term_.resize(running_cost_models_.size());
        for (std::size_t k = 0; k < running_cost_models_.size(); ++k) {
            const auto& cmap = running_cost_models_[k]->get_costs();
            for (const auto& kv : cmap) {
                names_set.insert(kv.first);
                node_has_term_[k][kv.first] = (kv.second && kv.second->active);
            }
        }
        if (include_terminal_ && terminal_cost_model_) {
            for (const auto& kv : terminal_cost_model_->get_costs()) names_set.insert(kv.first);
        }
        cost_names_.assign(names_set.begin(), names_set.end());

        terms_buffer_ << "iter,phase,k";
        for (const auto& name : cost_names_) terms_buffer_ << "," << name;
        terms_buffer_ << "\n";
    }

    void operator()(crocoddyl::SolverAbstract& solver) override {
        int it = solver.get_iter();

        // Console output
        // std::cout << "[CostRecorder] Iteration " << it << " cost=" << solver.get_cost() << std::endl;

        // Global stats
        stats_buffer_ << it << "," << solver.get_cost()
                      << "," << solver.get_steplength()
                      << "," << solver.get_stop()
                      << "," << solver.get_dVexp()
                      << "," << solver.get_dV() << "\n";

        // Per-node running costs
        for (std::size_t k = 0; k < running_cost_models_.size(); ++k) {
            terms_buffer_ << it << ",run," << k;
            for (const auto& name : cost_names_) {
                double v = 0.0;
                auto itp = node_has_term_[k].find(name);
                if (itp != node_has_term_[k].end() && itp->second) {
                    try { v = fetch_value_(name, static_cast<int>(k)); } catch (...) {}
                }
                terms_buffer_ << "," << v;
            }
            terms_buffer_ << "\n";
        }

        // Terminal cost
        if (include_terminal_ && terminal_cost_model_) {
            const int kterm = static_cast<int>(running_cost_models_.size());
            terms_buffer_ << it << ",term," << kterm;
            for (const auto& name : cost_names_) {
                double v = 0.0;
                const auto& cmap = terminal_cost_model_->get_costs();
                auto itc = cmap.find(name);
                if (itc != cmap.end() && itc->second && itc->second->active) {
                    try { v = fetch_value_(name, kterm); } catch (...) {}
                }
                terms_buffer_ << "," << v;
            }
            terms_buffer_ << "\n";
        }

        // Flush every iteration
        flushToFiles();
    }

private:
    void flushToFiles() {
        stats_.open(stats_csv_, std::ios::out | std::ios::trunc);
        stats_ << stats_buffer_.str();
        stats_.close();

        terms_.open(terms_csv_, std::ios::out | std::ios::trunc);
        terms_ << terms_buffer_.str();
        terms_.close();
    }

    std::vector<std::shared_ptr<crocoddyl::CostModelSum>> running_cost_models_;
    std::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model_;
    std::function<double(const std::string&, int)> fetch_value_;
    bool include_terminal_;
    std::vector<std::string> cost_names_;
    std::vector<std::unordered_map<std::string, bool>> node_has_term_;
    std::stringstream stats_buffer_, terms_buffer_;
    std::string stats_csv_, terms_csv_;
    std::ofstream stats_, terms_;
};