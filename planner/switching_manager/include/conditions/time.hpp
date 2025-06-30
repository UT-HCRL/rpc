#pragma once

#include <conditions/base.hpp>

/**
 * @brief Condition that checks if a certain time threshold has elapsed.
 * 
 * This condition evaluates whether the current time in the context
 * has reached or exceeded a specified threshold. It can be configured
 * to update the threshold either to a new value or a fixed value that 
 * is doubled on each update.
 */
class TimeElapsedCondition : public TransitionCondition {
public:
    TimeElapsedCondition(std::string name, double threshold, bool fixed_update) 
        : threshold_(threshold), fixed_update_(fixed_update) {
        setName(name);
    }
    
    bool evaluate(const Context& ctx) const override {
        constexpr double epsilon = 1e-6;
        return (ctx.time + epsilon) >= threshold_;
    }

    void update(double new_threshold = 0.0) {
        if (new_threshold <= 0.0 && !fixed_update_) {
            throw std::invalid_argument("Threshold must be positive.");
        }
        else if(new_threshold != 0 && fixed_update_) {
            throw std::invalid_argument("Cannot update threshold when fixed_update is true.");
        }
        threshold_ = fixed_update_ ? 2 * threshold_ : new_threshold;
    }
    
    std::string getName() const override { return name_; }

private:
    double threshold_;
    bool fixed_update_;
};