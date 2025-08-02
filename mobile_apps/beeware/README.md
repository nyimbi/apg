# APG Workflow Manager - BeeWare Mobile App

A comprehensive mobile workflow management application built with BeeWare and Python, providing native mobile experiences across iOS, Android, macOS, Windows, and Linux.

## Overview

APG Workflow Manager is a powerful mobile application that provides complete workflow management capabilities. Built with BeeWare's Toga framework, it offers native performance while maintaining a pure Python codebase.

### Key Features

- **Complete Workflow Management**: Design, execute, and monitor workflows
- **Real-time Collaboration**: Live updates and team coordination
- **Task Management**: Comprehensive task assignment and tracking
- **Offline-first Architecture**: Full functionality without internet connection
- **Biometric Authentication**: Secure fingerprint and face recognition
- **Cross-platform Native**: Single codebase for all platforms
- **Real-time Synchronization**: Automatic data sync when online
- **File Management**: Upload, download, and manage workflow files
- **Push Notifications**: Real-time alerts and updates
- **Voice Commands**: Voice-activated workflow operations

## Architecture

### Technology Stack

- **Framework**: BeeWare Toga (Python native mobile framework)
- **Language**: Python 3.10+
- **Authentication**: JWT with biometric support
- **Data Storage**: SQLite (offline) + PostgreSQL (backend)
- **Real-time**: WebSockets for live updates
- **Security**: Encrypted local storage with keyring
- **HTTP Client**: httpx for async API communication

### Project Structure

```
src/apg_workflow_mobile/
├── app.py                 # Main application entry point
├── models/               # Data models and schemas
│   ├── user.py           # User and authentication models
│   ├── workflow.py       # Workflow and instance models
│   ├── task.py          # Task management models
│   ├── notification.py   # Notification models
│   └── app_state.py     # Application state management
├── services/            # Business logic services
│   ├── api_service.py   # HTTP API communication
│   ├── auth_service.py  # Authentication and security
│   ├── workflow_service.py  # Workflow operations
│   ├── task_service.py  # Task management
│   ├── notification_service.py  # Notifications
│   ├── offline_service.py  # Offline data management
│   ├── biometric_service.py  # Biometric authentication
│   └── sync_service.py  # Data synchronization
├── ui/                  # User interface components
│   ├── screens/         # Main application screens
│   ├── components/      # Reusable UI components
│   └── dialogs/         # Modal dialogs
└── utils/               # Utility functions
    ├── constants.py     # Application constants
    ├── exceptions.py    # Custom exceptions
    ├── logger.py        # Logging configuration
    ├── security.py     # Security utilities
    └── validators.py    # Data validation
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Development Setup

1. **Clone the repository**:
   ```bash
   cd /path/to/apg/mobile_apps/beeware
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Run in development mode**:
   ```bash
   briefcase dev
   ```

### Building for Production

#### iOS

```bash
briefcase create iOS
briefcase build iOS
briefcase package iOS
```

#### Android

```bash
briefcase create android
briefcase build android
briefcase package android
```

#### macOS

```bash
briefcase create macOS
briefcase build macOS
briefcase package macOS
```

#### Windows

```bash
briefcase create windows
briefcase build windows
briefcase package windows
```

#### Linux

```bash
briefcase create linux
briefcase build linux
briefcase package linux
```

### Environment Configuration

Create a `.env` file for development:

```bash
# API Configuration
APG_API_BASE_URL=http://localhost:5000
APG_DEV_MODE=true
APG_DEBUG=true

# Logging
APG_LOG_LEVEL=DEBUG
APG_VERBOSE=true

# Features
APG_MOCK_API=false
```

## Features in Detail

### Authentication & Security

- **Multi-factor Authentication**: Username/password + biometric
- **Biometric Support**: Fingerprint, Face ID, Voice recognition
- **Secure Storage**: Encrypted local data with device keyring
- **Token Management**: Automatic refresh and secure storage
- **Device Registration**: Unique device identification and management

### Workflow Management

- **Visual Workflow Designer**: Drag-and-drop workflow creation
- **Workflow Templates**: Pre-built templates for common processes
- **Version Control**: Track workflow changes and rollbacks
- **Execution Monitoring**: Real-time workflow execution tracking
- **Performance Analytics**: Workflow performance metrics and insights

### Task Management

- **Task Assignment**: Assign tasks to users or groups
- **Progress Tracking**: Real-time task progress updates
- **Due Date Management**: Automated reminders and escalations
- **Task Dependencies**: Configure task relationships and dependencies
- **Collaboration**: Comments, attachments, and team communication

### Offline Capabilities

- **Offline-first Design**: Full functionality without internet
- **Local Data Storage**: SQLite database for offline data
- **Intelligent Sync**: Automatic synchronization when online
- **Conflict Resolution**: Smart handling of data conflicts
- **Queue Management**: Queue operations for later sync

### Real-time Features

- **Live Updates**: Real-time workflow and task updates
- **Push Notifications**: Instant alerts and notifications
- **WebSocket Communication**: Efficient real-time data exchange
- **Presence Indicators**: See who's online and active
- **Collaborative Editing**: Real-time collaborative features

## API Integration

The mobile app integrates with the APG backend services:

### Authentication Endpoints

- `POST /auth/login` - User login
- `POST /auth/biometric-login` - Biometric authentication
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Workflow Endpoints

- `GET /workflows` - List workflows
- `POST /workflows` - Create workflow
- `PUT /workflows/{id}` - Update workflow
- `POST /workflows/{id}/execute` - Execute workflow
- `GET /workflow-instances/{id}` - Get execution instance

### Task Endpoints

- `GET /tasks` - List tasks
- `POST /tasks` - Create task
- `PUT /tasks/{id}` - Update task
- `POST /tasks/{id}/assign` - Assign task
- `POST /tasks/{id}/complete` - Complete task

### Real-time WebSocket

- `/ws/notifications` - Notification updates
- `/ws/workflows` - Workflow updates
- `/ws/tasks` - Task updates
- `/ws/chat` - Team communication

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/apg_workflow_mobile

# Run specific test file
pytest tests/test_auth_service.py

# Run in verbose mode
pytest -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── ui/            # UI tests
└── fixtures/      # Test fixtures and data
```

## Deployment

### App Store Deployment (iOS)

1. Configure Apple Developer account in `pyproject.toml`
2. Build and sign the app:
   ```bash
   briefcase package iOS
   ```
3. Upload to App Store Connect

### Google Play Deployment (Android)

1. Configure Google Play signing in `pyproject.toml`
2. Build signed APK:
   ```bash
   briefcase package android
   ```
3. Upload to Google Play Console

### Enterprise Deployment

For enterprise deployment, configure MDM settings and distribute via enterprise app stores or direct installation.

## Configuration

### Application Settings

Settings are managed through `pyproject.toml` and environment variables:

```toml
[tool.briefcase.app.apg_workflow_mobile]
formal_name = "APG Workflow Manager"
bundle = "co.ke.datacraft"
version = "1.0.0"

# Platform-specific permissions
permission.INTERNET = "Access APG backend services"
permission.CAMERA = "Scan QR codes and capture images"
permission.USE_BIOMETRIC = "Biometric authentication"
```

### Feature Flags

Enable/disable features through configuration:

```python
FEATURES = {
    "biometric_auth": True,
    "offline_mode": True,
    "push_notifications": True,
    "file_upload": True,
    "voice_commands": True,
    "real_time_sync": True,
}
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow coding standards**: Use Black, isort, and mypy
4. **Write tests**: Ensure comprehensive test coverage
5. **Update documentation**: Keep README and docs current
6. **Submit pull request**: Provide detailed description

### Code Style

- **Formatting**: Black with 88-character line length
- **Import sorting**: isort with Black profile
- **Type checking**: mypy with strict settings
- **Docstrings**: Google-style docstrings
- **Comments**: Clear and concise inline comments

## Security Considerations

### Data Protection

- **Encryption at Rest**: All local data encrypted with device keys
- **Encryption in Transit**: TLS 1.3 for all API communication
- **Token Security**: Secure token storage with automatic refresh
- **Biometric Security**: Local biometric data never leaves device

### Privacy

- **Data Minimization**: Only collect necessary data
- **User Consent**: Clear consent for data collection
- **Data Retention**: Automatic cleanup of old data
- **Compliance**: GDPR and CCPA compliant

## Performance Optimization

### App Performance

- **Lazy Loading**: Load UI components on demand
- **Image Optimization**: Automatic image compression and caching
- **Memory Management**: Efficient memory usage and cleanup
- **Battery Optimization**: Optimized for battery life

### Network Performance

- **Request Batching**: Batch multiple API requests
- **Caching Strategy**: Intelligent caching of API responses
- **Compression**: Gzip compression for API requests
- **Connection Pooling**: Reuse HTTP connections

## Troubleshooting

### Common Issues

**App won't start**:
- Check Python version (3.10+ required)
- Verify all dependencies installed
- Check log files in `~/.apg_workflow_mobile/logs/`

**Authentication fails**:
- Verify API endpoint configuration
- Check network connectivity
- Clear stored credentials and retry

**Sync issues**:
- Check network connection
- Verify API authentication
- Review sync logs for errors

**Biometric not working**:
- Ensure device supports biometric authentication
- Check app permissions
- Re-enroll biometric data

### Debug Mode

Enable debug mode for detailed logging:

```bash
export APG_DEBUG=true
export APG_VERBOSE=true
briefcase dev
```

## License

Copyright © 2025 Datacraft. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or modification is strictly prohibited.

## Support

For technical support and questions:

- **Email**: support@datacraft.co.ke
- **Documentation**: https://docs.datacraft.co.ke/apg
- **Issue Tracker**: https://github.com/datacraft/apg/issues

## Changelog

### Version 1.0.0 (2025-01-XX)

**Initial Release**

- Complete workflow management functionality
- Cross-platform native mobile applications
- Biometric authentication support
- Offline-first architecture with sync
- Real-time collaboration features
- Comprehensive task management
- File upload/download capabilities
- Push notification system
- Voice command integration
- Enterprise security features