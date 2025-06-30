#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <Eigen/Geometry>


/**
 * @brief Manager for contact switching logic.
 * 
 * This class manages the contact switching logic based on the current
 * contact status and the number of knots in the MPC context.
 */
class ContactSwitchingManager {
public:
    ContactSwitchingManager(std::unordered_map<std::string, bool> contacts, int knots) 
        : frame_count_(contacts.size()), knots_(knots) {
        for (const auto& contact : contacts) {
            contact_status_[contact.first] = contact.second;
        }
        switching_mask_.resize(knots_, false);
    }

    void setSwitchingMask(int start_knot) {
        if (start_knot < 0 || start_knot >= knots_) {
            throw std::out_of_range("Knot index out of range.");
        }
        
        for (int i = start_knot; i < knots_; ++i) {
            switching_mask_[i] = true;
        }
    }

    void setSwitchingMaskRange(int start_knot, int end_knot) {
        if (start_knot < 0 || start_knot >= knots_ || end_knot < 0 || end_knot >= knots_) {
            throw std::out_of_range("Knot index out of range.");
        }
        
        for (int i = start_knot; i <= end_knot; ++i) {
            switching_mask_[i] = true;
        }
    }

    std::vector<bool> getSwitchingMask() const { return switching_mask_; }

    std::vector<bool> popSwitchingMask() {
        std::vector<bool> mask = switching_mask_;
        switching_mask_.assign(knots_, false); // Reset mask after popping
        return mask;
    }

    int getKnots() const { return knots_; }
    
    void resetSwitchingMask() {
        switching_mask_.assign(knots_, false);
    }

    void setContactStatus(const std::string& contact_name, bool active) {
        contact_status_[contact_name] = active;
    }

    bool getContactStatus(const std::string& contact_name) const {
        auto it = contact_status_.find(contact_name);
        return it != contact_status_.end() ? it->second : false;
    }

private:
    std::unordered_map<std::string, bool> contact_status_;
    std::vector<bool> switching_mask_;
    size_t frame_count_;
    const int knots_;
};