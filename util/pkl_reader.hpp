#pragma once

#include <string>
#include <fstream>
#include <variant>
#include <vector>
#include <unordered_map>
#include <memory>

#include "pkl_utils.hpp"

namespace pkl_utils{

    class __attribute__((visibility("hidden"))) PickleReader {
    public:

        explicit PickleReader(const std::string& filename = "", const PickleType& pkl_type = PickleType::DICT);
        ~PickleReader();

        /**
         * @brief Check if the file is ready to be parsed
         * @return true if the file is ready
         * @return false if the file is not ready
         * 
         */
        bool isReady() const;

        /**
         * @brief Parse the pickle file
         * 
         * This function parses the pickle file and extracts the data into bezier classes if PickleType is set to BEZIER.
         * If the PickleType is set to LIST or DICT, it will parse the data and print it.
         */
        void parse();

        /**
         * @brief Get the composite Bezier curves
         * @return std::vector<CompositeBezierCurve> vector of composite Bezier curves
         * 
         * This function returns the composite Bezier curves that were parsed from the pickle file.
         */
        std::vector<CompositeBezierCurve> getCompositeBezierCurves() const;

        const PickleType& getPickleType() const;

    private:

        class Impl;
        std::unique_ptr<Impl> impl_;
    
    };

}
