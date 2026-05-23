#include "safety_layer.h"

// Hard limits in degrees for each logical joint:
// J0: Base Yaw, J1: Shoulder (coupled), J2: Elbow/Wrist, J3: Gripper percentage
float JOINT_MIN[4] = {-150.0f, -90.0f, -120.0f,   0.0f};
float JOINT_MAX[4] = { 150.0f,  90.0f,   30.0f, 100.0f};

void safety_init() {
    // Initialise safety parameters, can be expanded to load dynamic calibration overrides from non-volatile flash (NVS).
}

void safety_clamp(float* joints, bool* clamped) {
    *clamped = false;
    for (int i = 0; i < 4; i++) {
        if (joints[i] < JOINT_MIN[i]) {
            joints[i] = JOINT_MIN[i];
            *clamped = true;
        }
        if (joints[i] > JOINT_MAX[i]) {
            joints[i] = JOINT_MAX[i];
            *clamped = true;
        }
    }
}
