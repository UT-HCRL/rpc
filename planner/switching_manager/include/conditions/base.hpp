#pragma once

#include <string>

#include "contact_switching/cs_utils.hpp"

using namespace ContactSwitchUtils;

/**
 * @brief Base class for transition conditions.
 * 
 * This class defines the interface for transition conditions that can be evaluated
 * against a context. Derived classes must implement the evaluate method to check
 * if the condition is met based on the provided context.
 */
class TransitionCondition {
public:
    virtual bool evaluate(const Context& ctx) const = 0;
    virtual ~TransitionCondition() = default;
    virtual std::string getName() const = 0;

protected:
    std::string name_;
    void setName(const std::string& name) { name_ = name; }
};
