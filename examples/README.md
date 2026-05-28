# APG Vision & IoT Example Application

A comprehensive Flask-AppBuilder application showcasing the powerful combination of computer vision and IoT management capabilities built with the APG framework.

Additional generated APG demonstration outputs live under `examples/generated/`.
Those files are retained as runnable/reference artifacts but are kept out of the
repository root so root-level entrypoints remain easy to scan.

## Parseable APG Language Examples

The numbered directories below contain fully annotated `main.apg` scripts. They are ordered from minimal record apps to larger ERP platform composition examples, and each script is parseable and compiler-clean with the current APG toolchain.

1. `01_minimal_customer_records/` - one table and module metadata.
2. `02_customer_orders_relationship/` - two related record entities.
3. `03_inventory_typed_records/` - typed integer and boolean fields.
4. `04_order_fulfillment_model/` - customer, order, shipment, and event records.
5. `05_single_support_agent/` - one first-class AI agent.
6. `06_support_agent_team/` - two agents connected by a team flow.
7. `07_multi_runtime_agent_team/` - Codex, Claude Code, OpenCode, OpenAI, Ollama, and Pi runtime declarations.
8. `08_basic_capability_contract/` - smallest capability contract.
9. `09_capability_rules_configuration/` - rules, approvals, and configuration.
10. `10_themed_i18n_streaming_capability/` - UI route, theme tokens, language codes, and ByteWax streaming.
11. `11_screen_composition_relationships/` - screen composition and element relationships.
12. `12_finance_general_ledger/` - finance ERP capability.
13. `13_procurement_approval_workbench/` - procurement approvals.
14. `14_inventory_warehouse_operations/` - inventory records plus warehouse capability.
15. `15_manufacturing_quality_control/` - production and quality-control capability.
16. `16_hr_payroll_operations/` - HR/payroll records, rules, approvals, and i18n.
17. `17_crm_sales_pipeline/` - CRM records and pipeline capability.
18. `18_operations_dashboard_capability/` - cross-functional operations dashboard.
19. `19_multi_capability_dependency_suite/` - dependency-planned capability suite.
20. `20_enterprise_erp_platform/` - larger ERP platform skeleton with records, capabilities, screens, rules, themes, i18n, and streaming.

## Features

### 🎯 **Computer Vision Capabilities**
- **Image Processing**: Object detection with face, person, vehicle, and general object recognition
- **Video Analysis**: Batch video processing with detection tracking and annotation
- **Live Streaming**: Real-time camera feed processing with live detection
- **Image Enhancement**: Automatic and manual image quality improvement
- **Analytics Dashboard**: Processing statistics and performance metrics

### 🌐 **IoT Management Capabilities**
- **Device Registration**: Easy device onboarding with comprehensive metadata
- **Sensor Data Collection**: Real-time sensor data recording and storage
- **Device Commands**: Remote device control and command execution
- **Alert System**: Intelligent monitoring with customizable alert rules
- **Data Visualization**: Real-time charts and analytics dashboards

### 📊 **Integrated Dashboard**
- **Unified Interface**: Single pane of glass for all operations
- **Real-time Monitoring**: Live system health and performance metrics
- **Activity Tracking**: Comprehensive logging of all operations
- **System Status**: Health monitoring with resource utilization
- **Quick Actions**: Easy access to common operations

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV-compatible camera (for live streaming)
- Modern web browser with JavaScript enabled

### Setup Instructions

1. **Clone the repository and navigate to examples:**
   ```bash
   cd /path/to/apg/examples
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python vision_iot_app.py
   ```

5. **Access the application:**
   - Open your browser to `http://localhost:8080`
   - Default admin login: `admin` / `admin`
   - Register new users via the registration link

## Application Structure

```
vision_iot_app/
├── vision_iot_app.py          # Main application file
├── requirements.txt           # Python dependencies
├── static/                    # Static files
│   ├── uploads/               # Upload directories
│   │   ├── cv/               # Computer vision uploads
│   │   └── iot/              # IoT data uploads
│   ├── outputs/              # Processing outputs
│   └── img/                  # Application images
├── templates/                 # Custom templates
│   ├── main_dashboard.html   # Main dashboard template
│   └── system_status.html    # System status template
└── vision_iot_app.db         # SQLite database (created on first run)
```

## Usage Guide

### Computer Vision Operations

#### Image Processing
1. Navigate to **AI & Vision > Computer Vision**
2. Click **Image Processing**
3. Upload an image file (JPG, PNG, BMP, TIFF)
4. Select detection types:
   - Face Detection
   - Person Detection
   - Vehicle Detection
   - Object Detection
5. Choose image enhancement (optional):
   - Auto Enhancement
   - Brightness/Contrast Adjustment
   - Noise Reduction
   - Sharpening
6. Click **Process Image**
7. View results with annotated detections

#### Video Processing
1. Navigate to **AI & Vision > Computer Vision**
2. Click **Video Processing**
3. Upload a video file (MP4, AVI, MOV, MKV, WebM)
4. Select detection types
5. Enable "Save Annotated Video" (optional)
6. Click **Process Video**
7. View frame-by-frame analysis results

#### Live Streaming
1. Navigate to **AI & Vision > Computer Vision**
2. Click **Live Stream**
3. Set camera ID (usually 0 for default camera)
4. Select detection types
5. Click **Start Live Stream**
6. View real-time detection in popup window
7. Use **Stop Stream** to terminate

### IoT Management Operations

#### Device Registration
1. Navigate to **IoT & Devices > IoT Management**
2. Click **Register Device**
3. Fill in device information:
   - **Device ID**: Unique identifier
   - **Name**: Human-readable name
   - **Type**: Sensor, Actuator, Gateway, etc.
   - **Connection**: WiFi, Bluetooth, Ethernet, etc.
   - **Location**: Physical location with GPS coordinates
   - **Sensors**: Primary sensor types
4. Click **Register Device**

#### Sensor Data Recording
1. Navigate to **IoT & Devices > IoT Management**
2. Click **Record Sensor Data**
3. Select target device
4. Enter sensor information:
   - **Sensor ID**: Unique sensor identifier
   - **Type**: Temperature, Humidity, Pressure, etc.
   - **Value**: Numeric reading
   - **Unit**: Measurement unit (°C, %, ppm, etc.)
   - **Quality**: Reading quality score (0-1)
5. Click **Record Data**

#### Device Commands
1. Navigate to **IoT & Devices > IoT Management**
2. Click **Send Commands**
3. Select target device
4. Enter command details:
   - **Command**: Command string
   - **Parameters**: JSON parameters (optional)
   - **Priority**: Command priority (1-10)
5. Click **Send Command**

#### Alert Rules
1. Navigate to **IoT & Devices > IoT Management**
2. Click **Manage Alerts**
3. Create alert rule:
   - **Name**: Rule identifier
   - **Device**: Target device (optional, blank for all)
   - **Sensor Type**: Monitor specific sensor type
   - **Condition**: Alert condition (e.g., "value > 30")
   - **Action**: Log, Email, Webhook, Command
4. Click **Create Rule**

### Dashboard and Monitoring

#### Main Dashboard
- **Overview**: System-wide statistics and health
- **Quick Actions**: Direct access to common operations
- **Recent Activity**: Latest processing and device activities
- **System Alerts**: Active alerts requiring attention

#### System Status
- **Health Score**: Overall system health percentage
- **Resource Metrics**: CPU, Memory, and Disk usage
- **Service Status**: Core and AI/IoT service status
- **Error Tracking**: Recent error counts and trends

#### Processing Logs
- Navigate to **Monitoring > Processing Logs**
- View all computer vision and IoT operations
- Filter by operation type, status, and date
- Detailed view with parameters and results

#### Device Activities
- Navigate to **Monitoring > Device Activities**
- Track all device-related events
- Filter by device, activity type, and severity
- Mark activities as resolved

## API Endpoints

The application provides REST API endpoints for integration:

### Computer Vision API
- `POST /computer_vision/api/process_image` - Process uploaded image
- `GET /computer_vision/analytics` - Get processing analytics

### IoT Management API  
- `GET /iot_management/api/sensor_data/<device_id>` - Get sensor readings
- `GET /iot_management/api/device_status/<device_id>` - Get device status

### Example API Usage
```bash
# Process image via API
curl -X POST http://localhost:8080/computer_vision/api/process_image \
  -F "image=@test_image.jpg" \
  -F "detection_types=face,person"

# Get sensor data
curl "http://localhost:8080/iot_management/api/sensor_data/device_001?hours=24&sensor_type=temperature"
```

## Configuration

### Application Settings
Edit `vision_iot_app.py` to modify:
- Database connection (default: SQLite)
- Upload file size limits
- Authentication settings
- Theme and appearance

### Security Configuration
- **Authentication**: Database-based user management
- **Authorization**: Role-based access control
- **File Upload**: Size and type restrictions
- **API Access**: Authenticated endpoints only

### Performance Tuning
- **Database**: Upgrade to PostgreSQL for production
- **File Storage**: Use cloud storage for large files
- **Caching**: Enable Redis for session management
- **Load Balancing**: Use Gunicorn with multiple workers

## Troubleshooting

### Common Issues

1. **Camera Access Failed**
   - Check camera permissions
   - Verify camera is not used by another application
   - Try different camera IDs (0, 1, 2, etc.)

2. **Upload Errors**
   - Check file size limits (default: 16MB)
   - Verify upload directory permissions
   - Ensure sufficient disk space

3. **Processing Failures**
   - Check system resources (CPU, Memory)
   - Verify OpenCV installation
   - Review processing logs for errors

4. **Database Issues**
   - Check SQLite file permissions
   - Verify database schema is current
   - Consider upgrading to PostgreSQL

### Performance Optimization

1. **Image Processing**
   - Resize large images before processing
   - Use appropriate detection types
   - Process videos in smaller chunks

2. **IoT Data**
   - Batch sensor data recordings
   - Archive old data regularly
   - Use appropriate alert thresholds

3. **System Resources**
   - Monitor CPU and memory usage
   - Adjust worker processes for load
   - Implement data retention policies

### Getting Help

- **Logs**: Check application logs in console output
- **Debug Mode**: Enable debug=True for detailed errors
- **System Status**: Use built-in health monitoring
- **API Testing**: Use /docs endpoint for API documentation

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 vision_iot_app:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "vision_iot_app:app"]
```

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
export DATABASE_URL=postgresql://user:pass@host:port/dbname
```

## Contributing

This example application demonstrates the power and flexibility of the APG framework. To contribute:

1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Submit pull requests

## License

This example application is provided under the same license as the APG framework. See the main project LICENSE file for details.

---

**APG Framework** - Empowering developers to build sophisticated applications with computer vision and IoT capabilities through intuitive, reusable components.
