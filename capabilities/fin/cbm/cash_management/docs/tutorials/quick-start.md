# APG Cash Management - Quick Start Tutorial

**Get Started with World-Class Cash Management in 15 Minutes**

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect

---

## 🚀 Welcome to APG Cash Management

This quick start guide will get you up and running with the APG Cash Management System in just 15 minutes. By the end of this tutorial, you'll have a fully functional cash management system with AI-powered forecasting, real-time bank integration, and executive dashboards.

### 📋 What You'll Learn

- **Installation**: Set up APG Cash Management locally
- **Configuration**: Connect your first bank account  
- **Data Import**: Load sample financial data
- **Forecasting**: Generate your first AI cash flow forecast
- **Dashboards**: View executive analytics and insights
- **API Usage**: Make your first API calls

### ⏱️ Estimated Time: 15 minutes

---

## 📦 Prerequisites

Before we begin, ensure you have:

- **Python 3.11+** installed
- **Docker & Docker Compose** (recommended)
- **Git** for cloning the repository
- **Basic command line knowledge**
- **Web browser** for the dashboard interface

### Quick Verification

```bash
# Verify prerequisites
python3 --version    # Should be 3.11+
docker --version     # Any recent version
git --version        # Any recent version
```

---

## 🎯 Step 1: Installation (3 minutes)

### Option A: Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/datacraft/apg-cash-management.git
cd apg-cash-management

# Start with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Option B: Local Installation

```bash
# Clone the repository
git clone https://github.com/datacraft/apg-cash-management.git
cd apg-cash-management

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env

# Initialize database
python manage.py migrate

# Start the application
python manage.py runserver
```

### ✅ Verification

Navigate to `http://localhost:8000/health` - you should see:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-27T10:00:00Z"
}
```

---

## 🔐 Step 2: Initial Setup (2 minutes)

### Create Your Admin User

```bash
# Create superuser account
python manage.py createsuperuser

# Follow the prompts:
# Username: admin
# Email: admin@yourcompany.com
# Password: [secure password]
```

### Access the Admin Interface

1. Navigate to `http://localhost:8000/admin`
2. Login with your admin credentials
3. You should see the APG Cash Management dashboard

### Quick Configuration

```bash
# Initialize sample tenant
python manage.py setup_sample_tenant

# This creates:
# - Sample tenant: "demo_company"
# - Sample user: demo@demo.com / demo123
# - Basic configuration
```

---

## 🏦 Step 3: Connect Your First Bank Account (3 minutes)

### Using the Web Interface

1. **Navigate to Cash Accounts**
   - Go to `http://localhost:8000/cash-management/accounts`
   - Click "Add New Account"

2. **Configure Account Details**
   ```
   Account Name: Main Checking Account
   Account Number: 1234567890
   Bank: Demo Bank (for testing)
   Account Type: Checking
   Currency: USD
   Initial Balance: $100,000.00
   ```

3. **Save Configuration**
   - Click "Save Account"
   - Account status should show "Active"

### Using the API (Alternative)

```bash
# Get authentication token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo@demo.com",
    "password": "demo123"
  }'

# Save the access_token from response
export ACCESS_TOKEN="your_access_token_here"

# Create bank account
curl -X POST http://localhost:8000/api/v1/accounts \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account_number": "1234567890",
    "account_type": "checking",
    "bank_id": "demo_bank",
    "currency": "USD",
    "initial_balance": 100000.00
  }'
```

### ✅ Verification

You should see your account in the accounts list with:
- ✅ Account Status: Active
- ✅ Current Balance: $100,000.00
- ✅ Last Updated: Just now

---

## 📊 Step 4: Import Sample Data (2 minutes)

### Load Sample Transactions

```bash
# Import sample cash flow data
python manage.py import_sample_data

# This loads:
# - 90 days of historical transactions
# - Various transaction categories
# - Realistic cash flow patterns
# - Customer and vendor payments
```

### Verify Data Import

```bash
# Check imported data
curl -X GET "http://localhost:8000/api/v1/cash-flows?limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

You should see sample transactions like:

```json
{
  "cash_flows": [
    {
      "id": "flow_1234567890",
      "amount": 15000.00,
      "transaction_date": "2025-01-27T09:30:00Z",
      "description": "Customer payment - Invoice #12345",
      "category": "operating_revenue",
      "counterparty": "ABC Corporation"
    }
  ]
}
```

---

## 🤖 Step 5: Generate Your First AI Forecast (3 minutes)

### Using the Web Interface

1. **Navigate to Forecasting**
   - Go to `http://localhost:8000/cash-management/forecasting`
   - Click "Generate New Forecast"

2. **Configure Forecast Parameters**
   ```
   Forecast Horizon: 30 days
   Confidence Level: 95%
   Model Type: Ensemble (Recommended)
   Include Scenarios: Yes
   ```

3. **Generate Forecast**
   - Click "Generate Forecast"
   - Wait 10-30 seconds for AI processing

### Using the API (Alternative)

```bash
# Generate forecast
curl -X POST http://localhost:8000/api/v1/forecasting/generate \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "acc_1234567890",
    "forecast_horizon": 30,
    "confidence_level": 0.95,
    "include_scenarios": true,
    "model_type": "ensemble"
  }'
```

### Understanding Your Forecast

The forecast will show:

- **📈 Daily Predictions**: Expected cash flows for next 30 days
- **🎯 Confidence Intervals**: Upper and lower bounds (95% confidence)
- **📊 Scenarios**: Optimistic, base case, and pessimistic scenarios
- **🎲 Model Performance**: Historical accuracy metrics

### Sample Forecast Results

```json
{
  "forecast_id": "forecast_1234567890",
  "model_used": "ensemble_v2.1",
  "predictions": [
    {
      "date": "2025-01-28",
      "predicted_amount": 12500.00,
      "confidence_lower": 10200.00,
      "confidence_upper": 14800.00
    }
  ],
  "scenarios": {
    "base_case": {"total_forecast": 385000.00},
    "optimistic": {"total_forecast": 425000.00},
    "pessimistic": {"total_forecast": 345000.00}
  },
  "model_performance": {
    "historical_accuracy": 0.94,
    "mean_absolute_error": 1250.00
  }
}
```

---

## 📈 Step 6: Explore Executive Dashboard (2 minutes)

### Access the Dashboard

Navigate to `http://localhost:8000/cash-management/dashboard`

### Key Dashboard Features

1. **💰 Cash Position Summary**
   - Total cash across all accounts
   - Available vs. restricted cash
   - 30-day change indicators

2. **📊 Account Breakdown**
   - Pie chart of account balances
   - Account type distribution
   - Performance metrics

3. **📈 Cash Flow Trends**
   - Historical cash flow chart
   - Inflow vs. outflow analysis
   - Trend indicators

4. **🔮 Forecast Visualization**
   - Interactive forecast chart
   - Confidence bands
   - Scenario comparison

5. **⚡ Real-Time Metrics**
   - Days cash on hand
   - Cash utilization rate
   - Liquidity ratios

### Dashboard API Access

```bash
# Get dashboard data
curl -X GET http://localhost:8000/api/v1/analytics/dashboard \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

---

## 🛡️ Step 7: Quick Risk Analysis

### Generate Risk Report

```bash
# Calculate risk metrics
curl -X POST http://localhost:8000/api/v1/risk/calculate \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "acc_1234567890": {
        "balance": 100000.00,
        "account_type": "checking"
      }
    },
    "risk_types": ["var", "liquidity"],
    "confidence_levels": [0.95],
    "time_horizons": [1]
  }'
```

### Risk Metrics Explained

- **📉 Value at Risk (VaR)**: Maximum expected loss over 1 day at 95% confidence
- **🌊 Liquidity Metrics**: Cash availability and funding stability
- **📊 Performance Ratios**: Risk-adjusted returns and volatility measures

---

## 🔧 Step 8: API Exploration

### Essential API Endpoints

```bash
# List all accounts
curl -X GET http://localhost:8000/api/v1/accounts \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Get recent cash flows
curl -X GET http://localhost:8000/api/v1/cash-flows?limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Get forecast performance
curl -X GET http://localhost:8000/api/v1/forecasting/performance \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Get system health
curl -X GET http://localhost:8000/health
```

### API Documentation

- **Interactive API Docs**: `http://localhost:8000/docs`
- **OpenAPI Specification**: `http://localhost:8000/openapi.json`
- **Redoc Documentation**: `http://localhost:8000/redoc`

---

## 🎉 Congratulations!

You've successfully set up APG Cash Management! Here's what you've accomplished:

✅ **Installed** the complete cash management system  
✅ **Connected** your first bank account  
✅ **Imported** sample financial data  
✅ **Generated** AI-powered cash flow forecasts  
✅ **Explored** executive dashboards and analytics  
✅ **Calculated** risk metrics and compliance reports  
✅ **Tested** REST API endpoints  

---

## 🚀 Next Steps

Now that you have the basics working, here are recommended next steps:

### 1. Real Bank Integration (15 minutes)
- [Bank API Integration Guide](../integration/bank-apis.md)
- Connect to Chase, Wells Fargo, or Bank of America
- Enable real-time transaction synchronization

### 2. Advanced Forecasting (10 minutes)
- [Advanced Forecasting Tutorial](advanced-forecasting.md)
- Configure multiple ML models
- Set up automated retraining

### 3. Risk Management (20 minutes)
- [Risk Analytics Setup](risk-setup.md)
- Configure VaR calculations
- Set up regulatory compliance reporting

### 4. Custom Optimization (15 minutes)
- [Custom Optimization Tutorial](custom-optimization.md)
- Multi-objective portfolio optimization
- Automated cash allocation recommendations

### 5. Production Deployment (30 minutes)
- [Production Deployment Guide](../deployment/production.md)
- Docker and Kubernetes deployment
- Security hardening and monitoring

---

## 📚 Additional Resources

### Documentation
- [API Reference](../api/README.md) - Complete API documentation
- [Architecture Guide](../architecture/system-overview.md) - System design and components
- [Security Guide](../admin/security.md) - Security best practices

### Video Tutorials
- [Getting Started Video](https://youtube.com/datacraft/apg-cash-management-intro) (10 minutes)
- [Advanced Features Demo](https://youtube.com/datacraft/apg-cash-management-advanced) (20 minutes)

### Community
- [GitHub Repository](https://github.com/datacraft/apg-cash-management)
- [Community Forum](https://community.datacraft.co.ke)
- [Support Portal](https://support.datacraft.co.ke)

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: "Connection refused" error  
**Solution**: Ensure Docker services are running: `docker-compose ps`

**Issue**: "Permission denied" on database  
**Solution**: Reset database: `docker-compose down -v && docker-compose up -d`

**Issue**: API returns 401 Unauthorized  
**Solution**: Refresh your access token using the `/auth/token` endpoint

**Issue**: Forecast generation fails  
**Solution**: Ensure you have at least 30 days of historical data

### Getting Help

- **Documentation**: Check the [troubleshooting guide](../admin/troubleshooting.md)
- **Community**: Post questions on our [forum](https://community.datacraft.co.ke)
- **Support**: Email support@datacraft.co.ke for technical assistance

---

## 🎯 Quick Reference

### Key URLs
- **Dashboard**: http://localhost:8000/cash-management/dashboard
- **API Docs**: http://localhost:8000/docs
- **Admin Panel**: http://localhost:8000/admin
- **Health Check**: http://localhost:8000/health

### Default Credentials
- **Admin**: admin / [your password]
- **Demo User**: demo@demo.com / demo123

### Important Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset everything
docker-compose down -v && docker-compose up -d
```

---

**Ready to transform your treasury operations?** 

Explore our [advanced tutorials](README.md) or jump into [production deployment](../deployment/production.md)!

---

*© 2025 Datacraft. All rights reserved.*