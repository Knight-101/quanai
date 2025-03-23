#!/bin/bash
# Run leverage analysis tool with commonly used options

# Default parameters
LOG_DIR="logs/incremental"
OUTPUT_DIR="analysis/leverage/$(date +%Y%m%d_%H%M%S)"
PLOT_STYLE="dark_background"

# Help function
function show_help {
    echo "Usage: ./run_leverage_analysis.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -l, --log-dir DIR         Set logs directory (default: logs/incremental)"
    echo "  -o, --output-dir DIR      Set output directory (default: analysis/leverage/TIMESTAMP)"
    echo "  -p, --phases PHASES       Analyze specific phases (space-separated numbers)"
    echo "  -s, --style STYLE         Set plot style (default: dark_background)"
    echo "                            Options: default, dark_background, ggplot, seaborn"
    echo ""
    echo "Examples:"
    echo "  ./run_leverage_analysis.sh"
    echo "  ./run_leverage_analysis.sh --phases 1 2 3"
    echo "  ./run_leverage_analysis.sh --log-dir logs/custom --style seaborn"
    exit 0
}

# Parse arguments
PHASES=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -p|--phases)
            shift
            PHASES=""
            # Collect all phase numbers until the next option
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                PHASES="$PHASES $1"
                shift
            done
            ;;
        -s|--style)
            PLOT_STYLE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python3 scripts/analyze_leverage.py --log-dir '$LOG_DIR' --output-dir '$OUTPUT_DIR' --plot-style '$PLOT_STYLE'"

if [ ! -z "$PHASES" ]; then
    CMD="$CMD --phases $PHASES"
fi

# Print the command and run it
echo "Running: $CMD"
eval $CMD

# Check if the analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Leverage analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Key files generated:"
    echo "  - leverage_summary.csv: Summary statistics for each phase"
    echo "  - leverage_by_phase.png: Box plot of leverage across phases"
    echo "  - leverage_metrics_heatmap.png: Heatmap of leverage metrics"
    echo ""
    echo "Phase-specific results are in subdirectories."
else
    echo "Leverage analysis failed. Check errors above."
fi 