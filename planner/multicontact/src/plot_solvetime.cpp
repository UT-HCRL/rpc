#include "humanoid_multicontact_tracker.hpp"
#include "util/util.hpp"
#include "fstream"
#include "mpc_utils.hpp"
#include "third_party/sciplot/sciplot.hpp"

using namespace sciplot;
using namespace mpc_utils;

// ----------- Utility plotting and stats -----------
template <typename T>
void plotFromVector(const std::vector<T>& data, std::string xlabel, std::string ylabel, std::string legend_label) {
    std::vector<double> iteration(data.size());
    std::iota(iteration.begin(), iteration.end(), 0.0);
    std::vector<double> data_as_double(data.begin(), data.end());

    Plot2D plot;
    plot.xlabel(xlabel);
    plot.ylabel(ylabel);
    plot.drawPoints(iteration, data_as_double).label(legend_label);

    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(750, 750);
    canvas.show();
}

void computeMean(const std::vector<double>& solve_times) {
    double sum = std::accumulate(solve_times.begin(), solve_times.end(), 0.0);
    double mean = sum / solve_times.size();
    std::cout << "Mean solve time: " << mean << " ms" << std::endl;
}

// ----------- File reading helpers -----------
std::vector<double> readSingleColumnFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + filename);
    }
    std::vector<double> values;
    double val;
    while (file >> val) {
        values.push_back(val);
    }
    return values;
}

MPCData parseBlock(const std::vector<std::string>& lines, int num_iter) {
    MPCData data;
    data.total_iterations = num_iter;
    std::string name;

    auto parseLine = [&](const std::string& line, std::vector<double>& container) {
        std::istringstream iss(line);
        for (int i = 0; i < num_iter; ++i) {
            double val; iss >> val;
            container.push_back(val);
        }
    };

    parseLine(lines[0], data.xReg_costs);
    parseLine(lines[1], data.uReg_costs);
    parseLine(lines[2], data.xBound_costs);
    parseLine(lines[3], data.com_costs);

    auto parseLabeledLine = [&](const std::string& line, auto& container) {
        std::istringstream iss(line);
        std::string name;
        iss >> name;
        if (!name.empty() && name.back() == ':') name.pop_back();
        for (int i = 0; i < num_iter; ++i) {
            double val; iss >> val;
            container[name].push_back(val);
        }
    };

    parseLabeledLine(lines[4], data.frame_costs);
    // parseLabeledLine(lines[5], data.contact_costs);
    // parseLabeledLine(lines[6], data.contact_costs);

    return data;
}

std::vector<MPCData> readMPCLogFile(const std::string& filename, int num_iter = 2) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + filename);
    }

    std::vector<MPCData> result;
    std::string line;
    while (true) {
        std::vector<std::string> block;
        for (int i = 0; i < 8 && std::getline(file, line); ++i) {
            block.push_back(line);
        }
        if (block.size() < 8) break;
        if (block[7] != "----------------------------------------") continue;

        result.push_back(parseBlock(block, num_iter));
    }

    return result;
}

void printMPCData(const std::vector<MPCData>& data_out) {
    for (const auto& data : data_out) {
        std::cout << "Total Iterations: " << data.total_iterations << "\n";

        auto printVec = [](const std::string& label, const std::vector<double>& vec) {
            std::cout << label << ": ";
            for (const auto& val : vec) std::cout << val << " ";
            std::cout << "\n";
        };

        printVec("xReg Costs", data.xReg_costs);
        printVec("uReg Costs", data.uReg_costs);
        printVec("xBound Costs", data.xBound_costs);
        printVec("COM Costs", data.com_costs);

        std::cout << "Frame Costs:\n";
        for (const auto& [key, values] : data.frame_costs) {
            std::cout << "  " << key << ": ";
            for (const auto& v : values) std::cout << v << " ";
            std::cout << "\n";
        }

        // std::cout << "Contact Costs:\n";
        // for (const auto& [key, values] : data.contact_costs) {
        //     std::cout << "  " << key << ": ";
        //     for (const auto& v : values) std::cout << v << " ";
        //     std::cout << "\n";
        // }

        std::cout << "-----------------------------\n";
    }
}

void plotMPCDataCost(const std::vector<MPCData>& data_out, const std::string& cost_type, double time_step = 0.003) {
    Plot3D plot;
    plot.xlabel("Iteration");
    plot.ylabel("Time [s]");
    plot.zlabel("Cost");
    plot.palette("viridis");
    plot.grid().show();
    plot.ztics().show();

    std::vector<double> x, y, z;

    for (size_t t = 0; t < data_out.size(); ++t) {
        double current_time = t * time_step;
        const std::vector<double>* costs = nullptr;

        if (cost_type == "xReg") costs = &data_out[t].xReg_costs;
        else if (cost_type == "uReg") costs = &data_out[t].uReg_costs;
        else if (cost_type == "xBound") costs = &data_out[t].xBound_costs;
        else if (cost_type == "CoM") costs = &data_out[t].com_costs;
        else {
            std::cerr << "Unknown cost type: " << cost_type << std::endl;
            return;
        }

        for (size_t i = 0; i < costs->size(); ++i) {
            x.push_back(static_cast<double>(i));
            y.push_back(current_time);
            z.push_back((*costs)[i]);
        }
    }

    plot.border().clear();
    plot.border().bottomLeftFront();
    plot.border().bottomRightFront();
    plot.border().leftVertical();
    plot.border().topLeftBack();
    plot.border().topRightBack();
    plot.border().rightVertical();

    double max_z = *std::max_element(z.begin(), z.end());
    plot.zrange(0.0, max_z);
    plot.autoclean(false);

    plot.drawPoints(x, y, z).label(cost_type + " Cost");
    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(1200, 1200);
    canvas.show();
}


// ----------- Main -----------
int main() {
    try {
        std::cout << "---\nEXAMPLE START---\n\n";

        auto iterations_per_solve = readSingleColumnFile("solve_iteration_log_perturbation.txt");
        auto solve_times = readSingleColumnFile("solve_timing_log_noperturbation.txt");
        auto data_out = readMPCLogFile("data_out_log.txt");

        printMPCData(data_out);
        plotMPCDataCost(data_out,"xReg");
        plotMPCDataCost(data_out,"uReg");
        plotMPCDataCost(data_out,"xBound");
        plotMPCDataCost(data_out,"CoM");

        // plotFromVector(iterations_per_solve, "iterations", "time [ms]", "solve time");
        // plotFromVector(solve_times, "iteration", "time [ms]", "solve time");
        // computeMean(solve_times);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
