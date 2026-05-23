#pragma once
#include <Arduino.h>

void contact_oracle_init(float threshold_dps);
void contact_oracle_push(float gx, float gy, float gz);  // Called for EVERY gyro sample from FIFO
bool contact_oracle_triggered();                          // Latched true after RMS threshold
bool contact_oracle_event();                              // True for one 50Hz cycle on rising edge
float contact_oracle_rms();                               // Current RMS value
void contact_oracle_reset();                              // Call when skill transitions to REACH
