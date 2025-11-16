#!/bin/bash
# hardware_verification.sh

echo "=== Hardware Authenticity Check ==="
echo ""

# Get serial number
SERIAL=$(system_profiler SPHardwareDataType | grep "Serial Number" | awk '{print $4}')
echo "Serial Number: $SERIAL"
echo "Check at: https://checkcoverage.apple.com"
echo ""

# Check if Find My is disabled (should be for resale)
echo "--- Activation Lock Status ---"
sudo profiles status -type enrollment 2>/dev/null

# Verify Apple Silicon chip authenticity
echo -e "\n--- Chip Information ---"
sysctl machdep.cpu.brand_string
system_profiler SPHardwareDataType | grep -E "Chip|Model Identifier"

# Check for original Apple parts
echo -e "\n--- Battery Information (OEM Check) ---"
system_profiler SPPowerDataType | grep -E "Manufacturer|Serial Number|Model"

# Display information
echo -e "\n--- Display (OEM Check) ---"
system_profiler SPDisplaysDataType | grep -E "Display Type|Resolution|Retina"

# Storage (check for original Apple SSD)
echo -e "\n--- Storage Controller ---"
system_profiler SPNVMeDataType 2>/dev/null || system_profiler SPSerialATADataType

# Check system integrity
echo -e "\n--- System Integrity Protection ---"
csrutil status

# Check if macOS is genuine
echo -e "\n--- macOS Integrity ---"
system_profiler SPSoftwareDataType | grep -E "System Version|Kernel Version"


# Check all hardware components for authenticity markers
sudo system_profiler SPHardwareDataType SPMemoryDataType SPStorageDataType SPPowerDataType SPDisplaysDataType SPAudioDataType > hardware_full_report.txt

# Check for third-party components
echo "Checking for non-Apple components..."

# Battery manufacturer (should be Apple/SMP/Sunwoda/Simplo)
ioreg -brc AppleSmartBattery | grep -E "Manufacturer|BatterySerialNumber"

# Display manufacturer
ioreg -lw0 | grep -i "display-" | grep -i "edid"