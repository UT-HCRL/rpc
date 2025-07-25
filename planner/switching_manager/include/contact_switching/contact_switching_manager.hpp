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
    ContactSwitchingManager(const std::vector<std::string>& frame_names, int knots) 
        : frame_count_(frame_names.size()), knots_(knots), current_phase_(0) {
        frame_names_.resize(frame_names.size());
        frame_names_ = frame_names;
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

    std::unordered_map<std::string, bool> getContactsStatus() const {
        return contact_status_;
    }

    std::pair<std::vector<std::string>, std::vector<std::string>> getContactsLists() const {
        std::vector<std::string> active_frames;
        std::vector<std::string> inactive_frames;

        for (const auto& frame : frame_names_) {
            if (contact_status_.at(frame)) {
                active_frames.push_back(frame);
            } else {
                inactive_frames.push_back(frame);
            }
        }

        return {active_frames, inactive_frames};
    }

    bool isMaskNotEmpty() const {
        return std::any_of(switching_mask_.begin(), switching_mask_.end(), [](bool v) { return v; });
    }

    bool isMaskAllTrue() const {
        return std::all_of(switching_mask_.begin(), switching_mask_.end(), [](bool v) { return v; });
    }

    bool hasTrueCountGreaterThan(int count) const {
        int true_count = std::count(switching_mask_.begin(), switching_mask_.end(), true);
        return true_count >= count;
    }

    void addContactPhase(const std::vector<bool>& planned_contacts){
        if (contact_status_.empty()) {
            for (size_t i = 0; i < frame_names_.size(); ++i) {
                contact_status_[frame_names_[i]] = planned_contacts[i];
            }
        }
        if (planned_contacts.size() != frame_count_) {
            throw std::invalid_argument("Planned contacts size does not match frame count.");
        }

        planned_contacts_.push_back(planned_contacts);
    }

    std::vector<bool> getNextPlannedContacts() {
        current_phase_++;
        if (current_phase_ < planned_contacts_.size()) {
            for (size_t i = 0; i < frame_names_.size(); ++i) {
                contact_status_[frame_names_[i]] = planned_contacts_[current_phase_][i];
            }
            return planned_contacts_[current_phase_];
        } else {
            throw std::out_of_range("No more planned contacts available.");
        }
    }

    std::vector<bool> getCurrentPlannedContacts() const {
        if (current_phase_ < planned_contacts_.size()) {
            return planned_contacts_[current_phase_];
        } else {
            throw std::out_of_range("Current phase exceeds planned contacts size.");
        }
    }
        
private:
    std::vector<std::string> frame_names_;
    std::unordered_map<std::string, bool> contact_status_;
    std::vector<bool> switching_mask_;
    size_t frame_count_;
    const int knots_;
    int current_phase_;
    std::vector<std::vector<bool>> planned_contacts_;
};