#pragma once

// Joint limits in degrees (set from calibration)
extern float JOINT_MIN[4];
extern float JOINT_MAX[4];

void safety_init();

// Clamps all 4 logical joint values in-place. Sets clamped_flag if any were modified.
void safety_clamp(float* joints, bool* clamped_flag);
