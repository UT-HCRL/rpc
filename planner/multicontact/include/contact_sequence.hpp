#pragma once

#include <iostream>

class ContactSequence {
    public:
        ContactSequence(int contact_phases = 0,
                        int phases_knots = 0,
                        double time_per_phase = 0.0)
            : contact_phases_(contact_phases), phases_knots_(phases_knots), phases_durations_(time_per_phase) {}
    
        const int& get_contact_phases() const { return contact_phases_; }
        const int& get_phases_knots() const { return phases_knots_; }
        const double& get_phases_durations() const { return phases_durations_; }

        void set_contact_phases(int contact_phases) { this->contact_phases_ = contact_phases; }
        void set_phases_knots(int phases_knots) { this->phases_knots_ = phases_knots; }
        void set_phases_durations(double phases_durations) { this->phases_durations_ = phases_durations; }

        friend std::ostream& operator<<(std::ostream& os, const ContactSequence& cs) {
            os << "Contact Phases: " << cs.contact_phases_ << "\n";
            os << "Phases Knots: " << cs.phases_knots_ << "\n";
            os << "Phases Durations: " << cs.phases_durations_ << "\n";
            return os;
        }

    private:
        int contact_phases_;
        int phases_knots_;
        double phases_durations_;
};