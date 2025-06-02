#!/bin/bash

# Check what's available in pico-extras
echo "=== Checking pico-extras structure ==="
ls -la $PICO_EXTRAS_PATH/src/rp2_common/

echo -e "\n=== Checking pico_sleep specifically ==="
ls -la $PICO_EXTRAS_PATH/src/rp2_common/pico_sleep/ 2>/dev/null || echo "pico_sleep directory not found"

echo -e "\n=== Looking for sleep-related files ==="
find $PICO_EXTRAS_PATH -name "*sleep*" -type f 2>/dev/null

echo -e "\n=== Checking CMake files for sleep ==="
find $PICO_EXTRAS_PATH -name "*.cmake" -exec grep -l "sleep" {} \; 2>/dev/null

echo -e "\n=== Available pico-extras libraries ==="
find $PICO_EXTRAS_PATH -name "CMakeLists.txt" -exec grep -H "pico_add_library\|add_library" {} \; 2>/dev/null | grep -v ".git"

echo "=== Contents of pico_sleep CMakeLists.txt ==="
cat $PICO_EXTRAS_PATH/src/rp2_common/pico_sleep/CMakeLists.txt

echo -e "\n=== Contents of pico_sleep header ==="
head -20 $PICO_EXTRAS_PATH/src/rp2_common/pico_sleep/include/pico/sleep.h

echo -e "\n=== Check if pico_sleep is included in main CMakeLists ==="
grep -n "pico_sleep" $PICO_EXTRAS_PATH/src/rp2_common/CMakeLists.txt