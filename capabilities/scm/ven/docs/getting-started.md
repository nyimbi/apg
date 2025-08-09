# Quick Start Guide

Get up and running with APG Vendor Management in minutes! This guide will walk you through the basic setup and your first vendor management workflow.

## 🎯 What You'll Learn

- How to install and configure the system
- Create your first vendor record
- Record performance data
- Generate AI insights
- Access analytics dashboards

## ⚡ Prerequisites

Before starting, ensure you have:

- **Python 3.12+** installed
- **PostgreSQL 14+** running
- **Redis 6+** for caching (optional but recommended)
- **Git** for cloning the repository
- **Administrative access** to create databases

## 🚀 Installation

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd vendor_management

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Database Setup

```bash
# Create database
createdb apg_vendor_management

# Run schema creation
psql apg_vendor_management < database_schema.sql

# Verify tables were created
psql apg_vendor_management -c "\dt vm_*"
```

### Step 3: Configuration

```bash
# Create configuration file
cp config.example.py config.py

# Edit with your settings
nano config.py
```

**Minimum Configuration:**
```python
# config.py
DATABASE_URL = 'postgresql://localhost/apg_vendor_management'
SECRET_KEY = 'your-secret-key-here'
REDIS_URL = 'redis://localhost:6379/0'  # Optional
```

### Step 4: Start the Application

```bash
# Start development server
python app.py

# Or with Flask CLI
flask run --host=0.0.0.0 --port=5000
```

**Success!** 🎉 Your application should now be running at:
- **Web Interface**: http://localhost:5000/vendor_management/
- **API Documentation**: http://localhost:5000/api/v1/docs

## 🏢 Your First Vendor

Let's create your first vendor record and explore the key features.

### Method 1: Web Interface

1. **Navigate to Vendor Dashboard**
   - Open http://localhost:5000/vendor_management/
   - Click "Manage Vendors" in the navigation

2. **Create New Vendor**
   ```
   Vendor Code: ACME001
   Name: ACME Corporation
   Legal Name: ACME Corporation Inc.
   Vendor Type: Supplier
   Category: Technology
   Email: contact@acme.com
   Phone: +1-555-0123
   Website: https://acme.com
   Strategic Importance: Standard
   ```

3. **Save and View**
   - Click "Save" to create the vendor
   - You'll see the vendor in the list with initial AI scores

### Method 2: API

```bash
# Create vendor via API
curl -X POST http://localhost:5000/api/v1/vendor-management/vendors \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000000" \
  -H "X-User-ID: 00000000-0000-0000-0000-000000000000" \
  -d '{
    "vendor_code": "ACME001",
    "name": "ACME Corporation",
    "vendor_type": "supplier",
    "category": "technology",
    "email": "contact@acme.com",
    "phone": "+1-555-0123",
    "website": "https://acme.com",
    "strategic_importance": "standard"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "vendor-uuid-here",
    "vendor_code": "ACME001",
    "name": "ACME Corporation",
    "status": "active",
    "created_at": "2025-01-29T12:00:00Z"
  }
}
```

## 📊 Record Performance Data

Now let's add some performance data to see the AI in action.

### Web Interface

1. **Go to Vendor Detail Page**
   - Click on your vendor in the list
   - Navigate to "Performance" tab

2. **Add Performance Record**
   ```
   Measurement Period: Quarterly
   Overall Score: 85.5
   Quality Score: 90.0
   Delivery Score: 82.0
   Cost Score: 88.0
   Service Score: 85.0
   On-Time Delivery Rate: 95.0%
   Quality Rejection Rate: 2.5%
   ```

### API Method

```bash
# Record performance data
curl -X POST http://localhost:5000/api/v1/vendor-management/vendors/{vendor_id}/performance \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000000" \
  -d '{
    "measurement_period": "quarterly",
    "overall_score": 85.5,
    "quality_score": 90.0,
    "delivery_score": 82.0,
    "cost_score": 88.0,
    "service_score": 85.0,
    "on_time_delivery_rate": 95.0,
    "quality_rejection_rate": 2.5
  }'
```

## 🤖 Generate AI Insights

Let's see the AI intelligence engine in action!

### Generate Intelligence

**Web Interface:**
1. Go to vendor detail page
2. Click "AI Intelligence" tab  
3. Click "Generate Fresh Intelligence"

**API Method:**
```bash
# Generate AI intelligence
curl -X POST http://localhost:5000/api/v1/vendor-management/vendors/{vendor_id}/intelligence \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000000"
```

**Sample AI Response:**
```json
{
  "success": true,
  "data": {
    "confidence_score": 0.85,
    "behavior_patterns": [
      {
        "pattern_type": "performance",
        "pattern_name": "consistent_high_performance",
        "confidence": 0.9,
        "description": "Vendor consistently delivers high-quality results"
      }
    ],
    "predictive_insights": [
      {
        "insight_type": "performance_forecast",
        "prediction": "improvement",
        "confidence": 0.8,
        "time_horizon": 90,
        "description": "Expected 5% improvement in overall performance"
      }
    ]
  }
}
```

### Get Optimization Recommendations

```bash
# Get optimization plan
curl -X POST http://localhost:5000/api/v1/vendor-management/vendors/{vendor_id}/optimization \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000000" \
  -d '{
    "objectives": ["performance_improvement", "cost_reduction"]
  }'
```

## 📈 Explore Analytics

### Dashboard Overview

1. **Navigate to Dashboard**
   - Go to http://localhost:5000/vendor_management/
   - View the AI-powered dashboard

2. **Key Metrics You'll See**
   - Total vendors and their status distribution
   - Average performance scores
   - Risk distribution
   - Recent activities
   - Top performing vendors

### Analytics API

```bash
# Get comprehensive analytics
curl http://localhost:5000/api/v1/vendor-management/analytics \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000000"
```

**Sample Analytics Response:**
```json
{
  "success": true,
  "data": {
    "vendor_counts": {
      "total_vendors": 1,
      "active_vendors": 1,
      "preferred_vendors": 0,
      "strategic_partners": 0
    },
    "performance_metrics": {
      "avg_performance": 85.5,
      "avg_risk": 25.0,
      "top_performers": 1
    },
    "recent_activities": []
  }
}
```

## ✅ Quick Success Checklist

Verify your setup is working correctly:

- [ ] **Application starts without errors**
- [ ] **Web interface loads at localhost:5000**
- [ ] **Can create a vendor record**
- [ ] **Performance data can be recorded**
- [ ] **AI intelligence generates successfully**
- [ ] **Dashboard displays vendor data**
- [ ] **API endpoints respond correctly**

## 🎯 Next Steps

Congratulations! You now have a working APG Vendor Management system. Here's what to explore next:

### Immediate Actions
1. **[User Guide](user-guide.md)** - Learn all features in detail
2. **[API Reference](api-reference.md)** - Explore complete API capabilities
3. **[Configuration](configuration.md)** - Customize for your needs

### Advanced Features
1. **Risk Management** - Add risk assessments and mitigation plans
2. **Vendor Portal** - Enable vendor self-service capabilities
3. **Integrations** - Connect with ERP and procurement systems
4. **Reporting** - Generate comprehensive vendor reports

### Production Readiness
1. **[Security Guide](security.md)** - Implement production security
2. **[Deployment Guide](deployment.md)** - Deploy to production
3. **[Monitoring](monitoring.md)** - Set up monitoring and alerts

## 🆘 Troubleshooting

### Common Issues

**Application Won't Start**
```bash
# Check Python version
python --version  # Should be 3.12+

# Check database connection
psql apg_vendor_management -c "SELECT 1;"

# Check dependencies
pip check
```

**Database Connection Errors**
```bash
# Verify PostgreSQL is running
pg_isready

# Check database exists
psql -l | grep apg_vendor_management

# Verify schema was created
psql apg_vendor_management -c "\dt vm_*"
```

**API Returns 500 Errors**
```bash
# Check application logs
tail -f logs/vendor_management.log

# Verify configuration
python -c "from config import Config; print(Config.DATABASE_URL)"
```

### Getting Help

If you encounter issues:

1. **Check [Troubleshooting Guide](troubleshooting.md)**
2. **Review application logs**
3. **Verify configuration settings**
4. **Contact support**: nyimbi@gmail.com

## 🎉 Welcome to APG Vendor Management!

You're now ready to transform your vendor management processes with AI-powered insights and automation. The system will learn from your data and provide increasingly valuable recommendations over time.

**Pro Tip**: The more data you add (vendors, performance records, communications), the more accurate and valuable the AI insights become!

---

**Ready for more?** Continue with the [User Guide](user-guide.md) to unlock the full potential of your vendor management system. 🚀