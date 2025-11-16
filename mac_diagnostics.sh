#!/bin/bash

echo "=== M1 MacBook Pro Diagnostics ==="
echo ""

# System Info
echo "--- System Information ---"
system_profiler SPHardwareDataType
echo ""

# GPU info
echo "--- GPU Information ---"
system_profiler SPDisplaysDataType | grep -E "GPU|Cores|Metal"
echo ""

# Battery Info
echo "--- Battery Health ---"
system_profiler SPPowerDataType | grep -E "Cycle Count|Condition|Full Charge|Health"
pmset -g batt
echo ""

# CPU Info
echo "--- CPU Information ---"
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu
sysctl -n hw.physicalcpu
echo ""

# Storage Health
echo "--- Storage Information ---"
diskutil info / | grep -E "SMART|Volume Name|File System|Disk Size|Free Space"
echo ""

# Temperature Sensors
echo "--- Temperature Check ---"
sudo powermetrics --samplers smc -i1 -n1 | grep -E "CPU die temperature"