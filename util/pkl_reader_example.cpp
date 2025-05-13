#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <map>
#include <string>
#include "pkl_reader.hpp"
#include "util/util.hpp"

using namespace pkl_utils;

int main(int argc, char** argv) {

    std::string file_path = THIS_COM "data_example/g1_test.pkl";
    // PickleReader reader("../data_example/g1_knee_knocker.pkl", PickleType::LIST);
    PickleReader reader(file_path, PickleType::BEZIER);
    if (!reader.isReady()) {
        std::cerr << "Failed to open the file." << std::endl;
        return -1;
    }    

    reader.parse();

    if(reader.getPickleType() == PickleType::BEZIER){

        std::vector<CompositeBezierCurve> read_beziers = reader.getCompositeBezierCurves();

        // for (const auto& bezier : read_beziers) {
        //     std::cout << "Bezier curve with " << bezier.getN() << " segments and dimension " << bezier.getD() << std::endl;
        //     std::cout << "Start: " << bezier.getA() << ", End: " << bezier.getB() << ", Duration: " << bezier.getDuration() << std::endl;
        // }
        std::cout << "[MAIN] - Number of Composite Bezier Obejcts: " << read_beziers.size() << std::endl;

        std::cout << "[MAIN] - Select bezier to print..." << std::endl;
        int selected_composite = 0;
        std::cin >> selected_composite;
        if (selected_composite < 0 || selected_composite >= read_beziers.size()) {
            std::cerr << "[MAIN] - Invalid selection." << std::endl;
            return -1;
        }
        const auto selected_bezier = read_beziers[selected_composite];
        
        std::cout << "[MAIN] - Points of selected bezier curve:" << std::endl;
        for (const auto& bezier : selected_bezier.getBeziers()) {
            const auto& points = bezier.getPoints();
            for (const auto& point : points) {
                std::cout << point.transpose() << std::endl;
            }
        }
    
    }
    
    return 0;
}
