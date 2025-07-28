#!/bin/bash

# APG Time & Attendance Capability - Test Runner Script
# Comprehensive test execution with reporting and CI/CD support
# Copyright © 2025 Datacraft

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="APG Time & Attendance"
TEST_DIR="tests"
REPORTS_DIR="test-reports"
COVERAGE_MIN=85

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
$PROJECT_NAME Test Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -u, --unit              Run unit tests only
    -i, --integration       Run integration tests only
    -p, --performance       Run performance tests only
    -s, --security          Run security tests only
    -a, --all               Run all tests (default)
    -c, --coverage          Generate coverage report
    -f, --fast              Skip slow tests
    -v, --verbose           Verbose output
    -q, --quiet             Quiet output
    --ci                    CI mode (fail fast, minimal output)
    --parallel              Run tests in parallel
    --clean                 Clean previous test artifacts
    --lint                  Run linting before tests
    --type-check            Run type checking before tests

EXAMPLES:
    $0                      # Run all tests
    $0 -u -c               # Run unit tests with coverage
    $0 -i --verbose        # Run integration tests with verbose output
    $0 --ci                # Run in CI mode
    $0 --clean --all       # Clean and run all tests

EOF
}

# Function to setup test environment
setup_test_env() {
    print_status "Setting up test environment..."
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    # Check if virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "No virtual environment detected"
        if [[ -f "venv/bin/activate" ]]; then
            print_status "Activating virtual environment..."
            source venv/bin/activate
        elif [[ -f ".venv/bin/activate" ]]; then
            print_status "Activating virtual environment..."
            source .venv/bin/activate
        else
            print_error "No virtual environment found. Please create one with 'python -m venv venv'"
            exit 1
        fi
    fi
    
    # Install test dependencies
    print_status "Installing test dependencies..."
    pip install -q -r requirements.txt
    
    print_success "Test environment ready"
}

# Function to clean previous artifacts
clean_artifacts() {
    print_status "Cleaning previous test artifacts..."
    
    rm -rf htmlcov/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    rm -rf .coverage
    rm -rf coverage.xml
    rm -rf junit.xml
    rm -rf "$REPORTS_DIR"/*
    
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Artifacts cleaned"
}

# Function to run linting
run_linting() {
    print_status "Running code linting..."
    
    # Check if tools are available
    if command -v ruff &> /dev/null; then
        print_status "Running Ruff linter..."
        ruff check . --output-format=github || {
            print_error "Ruff linting failed"
            exit 1
        }
    elif command -v flake8 &> /dev/null; then
        print_status "Running Flake8..."
        flake8 . || {
            print_error "Flake8 linting failed"
            exit 1
        }
    else
        print_warning "No linter found (ruff or flake8)"
    fi
    
    print_success "Linting passed"
}

# Function to run type checking
run_type_check() {
    print_status "Running type checking..."
    
    if command -v pyright &> /dev/null; then
        print_status "Running Pyright..."
        pyright . || {
            print_error "Type checking failed"
            exit 1
        }
    elif command -v mypy &> /dev/null; then
        print_status "Running MyPy..."
        mypy . || {
            print_error "Type checking failed"
            exit 1
        }
    else
        print_warning "No type checker found (pyright or mypy)"
    fi
    
    print_success "Type checking passed"
}

# Function to run tests
run_tests() {
    local test_args="$1"
    local test_type="$2"
    
    print_status "Running $test_type tests..."
    
    # Build pytest command
    local pytest_cmd="pytest"
    
    # Add coverage if requested
    if [[ "$COVERAGE" == "true" ]]; then
        pytest_cmd="$pytest_cmd --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing"
    fi
    
    # Add verbosity
    if [[ "$VERBOSE" == "true" ]]; then
        pytest_cmd="$pytest_cmd -v"
    elif [[ "$QUIET" == "true" ]]; then
        pytest_cmd="$pytest_cmd -q"
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL" == "true" ]] && command -v pytest-xdist &> /dev/null; then
        pytest_cmd="$pytest_cmd -n auto"
    fi
    
    # Add CI mode options
    if [[ "$CI_MODE" == "true" ]]; then
        pytest_cmd="$pytest_cmd --tb=short --maxfail=5"
    fi
    
    # Add fast mode (skip slow tests)
    if [[ "$FAST" == "true" ]]; then
        pytest_cmd="$pytest_cmd -m 'not slow'"
    fi
    
    # Add test arguments
    pytest_cmd="$pytest_cmd $test_args"
    
    print_status "Executing: $pytest_cmd"
    
    # Run tests
    if eval "$pytest_cmd"; then
        print_success "$test_type tests passed"
        return 0
    else
        print_error "$test_type tests failed"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."
    
    # Create comprehensive report
    cat > "$REPORTS_DIR/test_summary.md" << EOF
# $PROJECT_NAME - Test Report

**Generated:** $(date)
**Environment:** $(python --version)
**Test Directory:** $TEST_DIR

## Test Results

EOF
    
    # Add coverage information if available
    if [[ -f "coverage.xml" ]]; then
        local coverage_percent=$(grep -o 'line-rate="[^"]*"' coverage.xml | head -1 | grep -o '[0-9.]*' | awk '{print $1 * 100}')
        echo "**Coverage:** ${coverage_percent}%" >> "$REPORTS_DIR/test_summary.md"
        echo "" >> "$REPORTS_DIR/test_summary.md"
    fi
    
    # Add JUnit results if available
    if [[ -f "junit.xml" ]]; then
        local test_count=$(grep -o 'tests="[^"]*"' junit.xml | head -1 | grep -o '[0-9]*')
        local failure_count=$(grep -o 'failures="[^"]*"' junit.xml | head -1 | grep -o '[0-9]*')
        local error_count=$(grep -o 'errors="[^"]*"' junit.xml | head -1 | grep -o '[0-9]*')
        
        echo "**Total Tests:** $test_count" >> "$REPORTS_DIR/test_summary.md"
        echo "**Failures:** $failure_count" >> "$REPORTS_DIR/test_summary.md"
        echo "**Errors:** $error_count" >> "$REPORTS_DIR/test_summary.md"
        echo "" >> "$REPORTS_DIR/test_summary.md"
    fi
    
    # Move artifacts to reports directory
    [[ -f "coverage.xml" ]] && cp coverage.xml "$REPORTS_DIR/"
    [[ -f "junit.xml" ]] && cp junit.xml "$REPORTS_DIR/"
    [[ -d "htmlcov" ]] && cp -r htmlcov "$REPORTS_DIR/"
    
    print_success "Test report generated in $REPORTS_DIR/"
}

# Parse command line arguments
UNIT_TESTS=false
INTEGRATION_TESTS=false
PERFORMANCE_TESTS=false
SECURITY_TESTS=false
ALL_TESTS=true
COVERAGE=false
FAST=false
VERBOSE=false
QUIET=false
CI_MODE=false
PARALLEL=false
CLEAN=false
LINT=false
TYPE_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--unit)
            UNIT_TESTS=true
            ALL_TESTS=false
            shift
            ;;
        -i|--integration)
            INTEGRATION_TESTS=true
            ALL_TESTS=false
            shift
            ;;
        -p|--performance)
            PERFORMANCE_TESTS=true
            ALL_TESTS=false
            shift
            ;;
        -s|--security)
            SECURITY_TESTS=true
            ALL_TESTS=false
            shift
            ;;
        -a|--all)
            ALL_TESTS=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -f|--fast)
            FAST=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --ci)
            CI_MODE=true
            QUIET=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --lint)
            LINT=true
            shift
            ;;
        --type-check)
            TYPE_CHECK=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting $PROJECT_NAME test suite..."
    
    # Setup environment
    setup_test_env
    
    # Clean artifacts if requested
    if [[ "$CLEAN" == "true" ]]; then
        clean_artifacts
    fi
    
    # Run linting if requested
    if [[ "$LINT" == "true" ]]; then
        run_linting
    fi
    
    # Run type checking if requested
    if [[ "$TYPE_CHECK" == "true" ]]; then
        run_type_check
    fi
    
    # Track test results
    local test_results=0
    
    # Run specific test types
    if [[ "$ALL_TESTS" == "true" ]]; then
        run_tests "$TEST_DIR" "All" || test_results=$?
    else
        if [[ "$UNIT_TESTS" == "true" ]]; then
            run_tests "$TEST_DIR/ci" "Unit" || test_results=$?
        fi
        
        if [[ "$INTEGRATION_TESTS" == "true" ]]; then
            run_tests "$TEST_DIR/integration" "Integration" || test_results=$?
        fi
        
        if [[ "$PERFORMANCE_TESTS" == "true" ]]; then
            run_tests "-m performance" "Performance" || test_results=$?
        fi
        
        if [[ "$SECURITY_TESTS" == "true" ]]; then
            run_tests "-m security" "Security" || test_results=$?
        fi
    fi
    
    # Generate report
    generate_report
    
    # Final status
    if [[ $test_results -eq 0 ]]; then
        print_success "All tests completed successfully!"
        
        # Check coverage threshold
        if [[ "$COVERAGE" == "true" ]] && [[ -f "coverage.xml" ]]; then
            local coverage_percent=$(grep -o 'line-rate="[^"]*"' coverage.xml | head -1 | grep -o '[0-9.]*' | awk '{print $1 * 100}')
            if (( $(echo "$coverage_percent < $COVERAGE_MIN" | bc -l) )); then
                print_warning "Coverage $coverage_percent% is below minimum threshold $COVERAGE_MIN%"
                exit 1
            fi
        fi
        
        exit 0
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Run main function
main "$@"