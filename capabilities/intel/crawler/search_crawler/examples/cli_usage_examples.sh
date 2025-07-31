#!/bin/bash
# Search Crawler CLI Usage Examples
# ==================================
# 
# Comprehensive examples showing how to use the Search Crawler CLI
# for various scenarios and use cases.
#
# Author: Nyimbi Odero
# Company: Datacraft (www.datacraft.co.ke)

set -e  # Exit on error

CLI_SCRIPT="../cli.py"
RESULTS_DIR="./cli_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "🔍 Search Crawler CLI Examples"
echo "=============================="

# Check CLI availability
echo ""
echo "📋 1. Health Check"
echo "==================="
python "$CLI_SCRIPT" --health-check

echo ""
echo "📋 2. List Available Engines"
echo "============================="
python "$CLI_SCRIPT" --list-engines

echo ""
echo "🔍 3. Basic Search Examples"
echo "==========================="

# Basic search with default settings
echo "Basic search with intelligent defaults:"
python "$CLI_SCRIPT" "Ethiopia news" --max-results 5 --output-format text

# Search with specific engines
echo ""
echo "Search with specific engines:"
python "$CLI_SCRIPT" "Somalia security" --engines google,bing,duckduckgo --max-results 3 --output-format table

# Save results to file
echo ""
echo "Save results to JSON file:"
python "$CLI_SCRIPT" "Horn of Africa drought" --max-results 5 --output-format json --output-file "$RESULTS_DIR/drought_results.json"

echo ""
echo "🚨 4. Conflict Monitoring Examples"
echo "=================================="

# Conflict monitoring mode
echo "Conflict monitoring with alerts:"
python "$CLI_SCRIPT" "Sudan conflict violence" --mode conflict --enable-alerts --max-results 5 --output-format text

# Geographic targeting
echo ""
echo "Geographic targeting:"
python "$CLI_SCRIPT" "Ethiopia civil unrest" --mode conflict --target-countries ET,SO,ER --max-results 3 --output-format table

# High escalation threshold
echo ""
echo "High-priority conflicts only:"
python "$CLI_SCRIPT" "Somalia Al-Shabaab attack" --mode conflict --escalation-threshold 0.9 --max-results 3 --output-format text

echo ""
echo "⚙️ 5. Performance Optimization Examples"
echo "======================================="

# Fast search
echo "Fast search with minimal resources:"
python "$CLI_SCRIPT" "quick search test" --max-results 5 --engines duckduckgo --timeout 15 --output-format text

# Comprehensive search
echo ""
echo "Comprehensive search with high quality:"
python "$CLI_SCRIPT" "detailed analysis test" --max-results 10 --min-relevance-score 0.8 --enable-quality-filtering --output-format table

echo ""
echo "📊 6. Output Format Examples"
echo "==========================="

# JSON output
echo "JSON format:"
python "$CLI_SCRIPT" "test query" --max-results 3 --output-format json --output-file "$RESULTS_DIR/json_example.json"
echo "Results saved to $RESULTS_DIR/json_example.json"

# CSV output
echo ""
echo "CSV format:"
python "$CLI_SCRIPT" "test query" --max-results 3 --output-format csv --output-file "$RESULTS_DIR/csv_example.csv"
echo "Results saved to $RESULTS_DIR/csv_example.csv"

# YAML output
echo ""
echo "YAML format:"
python "$CLI_SCRIPT" "test query" --max-results 3 --output-format yaml --output-file "$RESULTS_DIR/yaml_example.yaml"
echo "Results saved to $RESULTS_DIR/yaml_example.yaml"

echo ""
echo "⚙️ 7. Configuration Examples"
echo "============================"

# Save configuration
echo "Saving configuration:"
python "$CLI_SCRIPT" "config test" --mode conflict --engines google,bing --max-results 20 --save-config "$RESULTS_DIR/my_config.yaml" --output-format text

# Load configuration
echo ""
echo "Loading configuration:"
if [ -f "$RESULTS_DIR/my_config.yaml" ]; then
    python "$CLI_SCRIPT" --load-config "$RESULTS_DIR/my_config.yaml" --output-format text
else
    echo "Configuration file not found, creating example..."
    cat > "$RESULTS_DIR/example_config.yaml" << EOF
mode: "conflict"
query: "Horn of Africa crisis"
engines:
  - "google" 
  - "bing"
  - "duckduckgo"
max_results: 15
target_countries:
  - "ET"
  - "SO"
  - "ER"
enable_alerts: true
output_format: "json"
EOF
    echo "Example configuration created at $RESULTS_DIR/example_config.yaml"
fi

echo ""
echo "🛠️ 8. Advanced Features"
echo "======================="

# Performance benchmark
echo "Running performance benchmark:"
python "$CLI_SCRIPT" --benchmark || echo "Benchmark requires additional components"

# Verbose mode
echo ""
echo "Verbose mode with detailed logging:"
python "$CLI_SCRIPT" "verbose test" --max-results 3 --verbose --output-format text

echo ""
echo "✅ CLI Examples Complete"
echo "========================"
echo ""
echo "Results have been saved to: $RESULTS_DIR/"
echo ""
echo "To run interactive mode:"
echo "  python $CLI_SCRIPT --interactive"
echo ""
echo "To start monitoring mode:"
echo "  python $CLI_SCRIPT 'Ethiopia conflict' --monitor-mode --monitor-interval 300"
echo ""
echo "For help with any command:"
echo "  python $CLI_SCRIPT --help"
echo ""
echo "For more examples, see the configuration files in the configs/ directory."