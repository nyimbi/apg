#!/bin/bash
# APG Employee Data Management - Load Testing Script

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
API_HOST=${1:-http://localhost:8000}
USERS=${2:-50}
SPAWN_RATE=${3:-5}
DURATION=${4:-300}  # 5 minutes default
TEST_TYPE=${5:-normal}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if locust is installed
    if ! command -v locust &> /dev/null; then
        log_error "Locust is not installed. Installing..."
        pip install locust
    fi
    
    # Check if API is accessible
    if ! curl -f -s "$API_HOST/api/v1/health" > /dev/null; then
        log_warning "API health check failed. Proceeding anyway..."
    else
        log_success "API is accessible"
    fi
    
    # Create results directory
    mkdir -p "$PROJECT_DIR/load_test_results"
    
    log_success "Prerequisites check completed"
}

# Run basic load test
run_basic_load_test() {
    log_info "Running basic load test..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_dir="$PROJECT_DIR/load_test_results/basic_$timestamp"
    mkdir -p "$results_dir"
    
    cd "$PROJECT_DIR"
    
    locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="$USERS" \
        --spawn-rate="$SPAWN_RATE" \
        --run-time="${DURATION}s" \
        --headless \
        --html="$results_dir/report.html" \
        --csv="$results_dir/results" \
        --only-summary \
        EmployeeManagementUser
    
    log_success "Basic load test completed. Results saved to $results_dir"
}

# Run stress test
run_stress_test() {
    log_info "Running stress test with high volume users..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_dir="$PROJECT_DIR/load_test_results/stress_$timestamp"
    mkdir -p "$results_dir"
    
    cd "$PROJECT_DIR"
    
    # Gradually increase load
    local stages=(
        "10:30"   # 10 users for 30 seconds
        "25:60"   # 25 users for 60 seconds
        "50:120"  # 50 users for 2 minutes
        "100:180" # 100 users for 3 minutes
        "200:120" # 200 users for 2 minutes
        "100:60"  # Scale down to 100 for 1 minute
        "50:30"   # Scale down to 50 for 30 seconds
    )
    
    for stage in "${stages[@]}"; do
        IFS=':' read -r users duration <<< "$stage"
        log_info "Running stage: $users users for ${duration}s"
        
        timeout "${duration}s" locust \
            -f tests/load_test.py \
            --host="$API_HOST" \
            --users="$users" \
            --spawn-rate="$SPAWN_RATE" \
            --headless \
            --csv="$results_dir/stress_${users}users" \
            HighVolumeUser || true
        
        sleep 5  # Brief pause between stages
    done
    
    # Generate final report
    locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="$USERS" \
        --spawn-rate="$SPAWN_RATE" \
        --run-time="60s" \
        --headless \
        --html="$results_dir/stress_report.html" \
        HighVolumeUser
    
    log_success "Stress test completed. Results saved to $results_dir"
}

# Run soak test (long duration)
run_soak_test() {
    log_info "Running soak test (long duration)..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_dir="$PROJECT_DIR/load_test_results/soak_$timestamp"
    mkdir -p "$results_dir"
    
    cd "$PROJECT_DIR"
    
    # Run for 30 minutes with moderate load
    locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="30" \
        --spawn-rate="2" \
        --run-time="1800s" \
        --headless \
        --html="$results_dir/soak_report.html" \
        --csv="$results_dir/soak_results" \
        EmployeeManagementUser
    
    log_success "Soak test completed. Results saved to $results_dir"
}

# Run spike test
run_spike_test() {
    log_info "Running spike test..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_dir="$PROJECT_DIR/load_test_results/spike_$timestamp"
    mkdir -p "$results_dir"
    
    cd "$PROJECT_DIR"
    
    # Normal load
    log_info "Phase 1: Normal load (20 users)"
    timeout "60s" locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="20" \
        --spawn-rate="5" \
        --headless \
        --csv="$results_dir/spike_normal" \
        EmployeeManagementUser || true
    
    sleep 5
    
    # Spike
    log_info "Phase 2: Spike (500 users)"
    timeout "120s" locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="500" \
        --spawn-rate="50" \
        --headless \
        --csv="$results_dir/spike_peak" \
        HighVolumeUser || true
    
    sleep 5
    
    # Return to normal
    log_info "Phase 3: Return to normal (20 users)"
    locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="20" \
        --spawn-rate="5" \
        --run-time="60s" \
        --headless \
        --html="$results_dir/spike_report.html" \
        EmployeeManagementUser
    
    log_success "Spike test completed. Results saved to $results_dir"
}

# Run reporting focused test
run_reporting_test() {
    log_info "Running reporting and analytics focused test..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_dir="$PROJECT_DIR/load_test_results/reporting_$timestamp"
    mkdir -p "$results_dir"
    
    cd "$PROJECT_DIR"
    
    locust \
        -f tests/load_test.py \
        --host="$API_HOST" \
        --users="25" \
        --spawn-rate="3" \
        --run-time="${DURATION}s" \
        --headless \
        --html="$results_dir/reporting_report.html" \
        --csv="$results_dir/reporting_results" \
        ReportingUser
    
    log_success "Reporting test completed. Results saved to $results_dir"
}

# Monitor system resources during test
monitor_resources() {
    local results_dir="$1"
    local duration="$2"
    
    log_info "Starting resource monitoring..."
    
    # Monitor CPU, memory, and network
    {
        echo "timestamp,cpu_percent,memory_percent,disk_io,network_io"
        for ((i=1; i<=duration; i++)); do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            cpu=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' || echo "0")
            memory=$(vm_stat | awk '/Pages active:/ {print $3}' | sed 's/\.//' || echo "0")
            echo "$timestamp,$cpu,$memory,0,0"
            sleep 1
        done
    } > "$results_dir/system_metrics.csv" &
    
    local monitor_pid=$!
    echo $monitor_pid > "$results_dir/monitor.pid"
    
    log_info "Resource monitoring started (PID: $monitor_pid)"
}

# Generate summary report
generate_summary() {
    local results_base="$PROJECT_DIR/load_test_results"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local summary_file="$results_base/summary_$timestamp.md"
    
    log_info "Generating load test summary..."
    
    cat > "$summary_file" << EOF
# APG Employee Management Load Test Summary

**Date:** $(date)
**API Host:** $API_HOST
**Test Configuration:**
- Users: $USERS
- Spawn Rate: $SPAWN_RATE
- Duration: ${DURATION}s
- Test Type: $TEST_TYPE

## Test Results

### Key Metrics
- **Total Requests:** $(find "$results_base" -name "*.csv" -exec tail -1 {} \; | head -1 | cut -d',' -f3 || echo "N/A")
- **Failure Rate:** $(find "$results_base" -name "*_stats.csv" -exec tail -1 {} \; | head -1 | cut -d',' -f4 || echo "N/A")
- **Average Response Time:** $(find "$results_base" -name "*_stats.csv" -exec tail -1 {} \; | head -1 | cut -d',' -f6 || echo "N/A")ms

### Test Files Generated
$(find "$results_base" -name "*.html" -o -name "*.csv" | tail -10)

### Recommendations
1. Monitor response times under high load
2. Check for memory leaks during soak tests
3. Validate error handling during spike tests
4. Review database performance metrics
5. Consider implementing rate limiting if needed

### Next Steps
1. Review detailed HTML reports
2. Analyze system metrics
3. Compare with previous test runs
4. Optimize identified bottlenecks

---
Generated by APG Employee Management Load Testing Suite
EOF
    
    log_success "Summary report generated: $summary_file"
}

# Main execution
main() {
    echo "🚀 APG Employee Management Load Testing Suite"
    echo "=============================================="
    echo "API Host: $API_HOST"
    echo "Users: $USERS"
    echo "Spawn Rate: $SPAWN_RATE"
    echo "Duration: ${DURATION}s"
    echo "Test Type: $TEST_TYPE"
    echo ""
    
    check_prerequisites
    
    case "$TEST_TYPE" in
        "basic"|"normal")
            run_basic_load_test
            ;;
        "stress")
            run_stress_test
            ;;
        "soak")
            run_soak_test
            ;;
        "spike")
            run_spike_test
            ;;
        "reporting")
            run_reporting_test
            ;;
        "all")
            log_info "Running comprehensive test suite..."
            run_basic_load_test
            sleep 30
            run_stress_test
            sleep 30
            run_spike_test
            sleep 30
            run_reporting_test
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            echo "Available types: basic, stress, soak, spike, reporting, all"
            exit 1
            ;;
    esac
    
    generate_summary
    
    log_success "Load testing completed successfully!"
    echo ""
    echo "📊 Results available in: $PROJECT_DIR/load_test_results/"
    echo "🔍 Open the HTML reports in your browser for detailed analysis"
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [API_HOST] [USERS] [SPAWN_RATE] [DURATION] [TEST_TYPE]"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Basic test with defaults"
    echo "  $0 http://localhost:8000 100 10 600 stress # Stress test with 100 users"
    echo "  $0 https://api.company.com 50 5 300 all    # Full test suite"
    echo ""
    echo "Test Types: basic, stress, soak, spike, reporting, all"
    echo ""
    exit 1
fi

# Run main function
main "$@"