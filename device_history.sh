#!/bin/bash
# device_history.sh

echo "=== Device History Check ==="
echo ""

# Check system language/region history
echo "--- Current Region Settings ---"
defaults read -g AppleLocale
defaults read -g Country

# Check timezone history
echo -e "\n--- Timezone ---"
sudo systemsetup -gettimezone

# Check for previous user accounts
echo -e "\n--- User Accounts ---"
dscl . list /Users | grep -v '^_'

# Check system logs for location/timezone changes
echo -e "\n--- Recent Region Changes (if any) ---"
log show --predicate 'eventMessage contains "timezone" OR eventMessage contains "locale"' --info --last 30d | head -20

# Check keyboard layouts (can indicate regions)
echo -e "\n--- Keyboard Input Sources ---"
defaults read com.apple.HIToolbox AppleEnabledInputSources

# WiFi networks previously connected (requires sudo)
echo -e "\n--- Previous WiFi Networks ---"
sudo networksetup -listpreferredwirelessnetworks en0 2>/dev/null || echo "Unable to read WiFi history"