#!/bin/bash

# CPU Profiler Script - Linux perf profiling tool
# Usage: ./cpu_profiler.sh <profile_file> <output_file> <executable> [args...]

set -e  # Exit on any error

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <profile_file> <output_file> <executable> [args...]"
    echo ""
    echo "Arguments:"
    echo "  profile_file  - Output file for perf data (e.g., perf.data)"
    echo "  output_file   - Text report output file (e.g., report.txt)"
    echo "  executable    - Program to profile"
    echo "  args...       - Arguments for the executable"
    echo ""
    echo "Example:"
    echo "  $0 perf.data report.txt ./myapp arg1 arg2"
    exit 1
}

check_dependencies() {
    if ! command -v perf >/dev/null 2>&1; then
        echo -e "${RED}Error: perf not found. Install linux-tools package${NC}" >&2
        exit 1
    fi
}

profile_with_perf() {
    local profile_file=$1
    local output_file=$2
    local executable=$3
    shift 3
    
    echo -e "${BLUE}[INFO] Starting perf profiling...${NC}"
    
    # Record performance data
    echo -e "${BLUE}[INFO] Recording perf data...${NC}"
    perf record -g -o "$profile_file" "$executable" "$@"
    
    # Generate text report
    if [ -f "$profile_file" ]; then
        echo -e "${BLUE}[INFO] Generating perf report...${NC}"
        {
            echo "=== PERF STAT SUMMARY ==="
            perf stat -e cache-references,cache-misses,cycles,instructions,branches,branch-misses "$executable" "$@" 2>&1 || true
            echo ""
            echo "=== DETAILED PROFILE REPORT ==="
            perf report -i "$profile_file" --stdio
        } > "$output_file" 2>&1
        
        echo -e "${GREEN}[SUCCESS] Perf data: $profile_file${NC}"
        echo -e "${GREEN}[SUCCESS] Text report: $output_file${NC}"
        
        # Show quick summary
        echo -e "${BLUE}[INFO] Top functions:${NC}"
        perf report -i "$profile_file" --stdio | head -20
    else
        echo -e "${RED}[ERROR] Perf data file $profile_file was not created${NC}" >&2
        exit 1
    fi
}

# Main script
if [ "$#" -lt 3 ]; then
    print_usage
fi

PROFILE_FILE=$1
OUTPUT_FILE=$2
EXECUTABLE=$3
shift 3

# Validate inputs
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable '$EXECUTABLE' not found${NC}" >&2
    exit 1
fi

if [ ! -x "$EXECUTABLE" ]; then
    echo -e "${RED}Error: '$EXECUTABLE' is not executable${NC}" >&2
    exit 1
fi

# Check dependencies
check_dependencies

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Run perf profiling
profile_with_perf "$PROFILE_FILE" "$OUTPUT_FILE" "$EXECUTABLE" "$@"

echo -e "${GREEN}[DONE] Profiling completed successfully!${NC}"
