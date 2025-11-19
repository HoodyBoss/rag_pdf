# RAG PDF Authentication System

This document describes the authentication system implementation for the RAG PDF application.

## Overview

The authentication system provides secure login functionality with:
- JWT-based authentication
- MongoDB data storage
- Session management
- Rate limiting
- Activity logging
- Default admin account

## Security Features

- **Password Hashing**: Uses bcrypt for secure password storage
- **JWT Tokens**: Access and refresh tokens with configurable expiration
- **Rate Limiting**: Prevents brute force attacks
- **Activity Logging**: Security audit trail
- **Session Management**: Secure session handling
- **Role-based Access**: Admin and user roles

## Files Structure

```
rag_pdf/
├── auth_models.py          # Authentication backend (AuthManager class)
├── login_page.py          # Login UI (Gradio interface)
├── authenticated_app.py   # Main application wrapper
├── init_auth.py          # Database initialization script
├── test_auth.py          # Authentication testing script
└── AUTH_README.md        # This documentation
```

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=rag_pdf_auth

# Authentication Configuration
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
ADMIN_PASSWORD=admin123

# Other configurations...
```

### 2. MongoDB Setup

**Option A: Local MongoDB**
```bash
# Install and start MongoDB
# Windows: Download and install MongoDB Community Server
# Linux: sudo systemctl start mongod
# macOS: brew services start mongodb-community
```

**Option B: MongoDB Atlas (Cloud)**
1. Create account at https://www.mongodb.com/atlas
2. Create a free cluster
3. Get connection string
4. Update `MONGODB_URI` in `.env`

### 3. Install Dependencies

```bash
pip install -r requirements_minimal.txt
```

### 4. Initialize Database

```bash
python init_auth.py
```

This will:
- Connect to MongoDB
- Create necessary collections and indexes
- Create default admin account (username: admin, password: from ADMIN_PASSWORD env)

### 5. Test Authentication

```bash
python test_auth.py
```

### 6. Run Authenticated Application

```bash
python authenticated_app.py
```

## Default Credentials

- **Username**: `admin`
- **Password**: `admin123` (or ADMIN_PASSWORD environment variable)
- **Role**: `admin`

⚠️ **IMPORTANT**: Change the default admin password immediately after first login!

## Authentication Flow

1. **Login**: User enters credentials on login page
2. **Validation**: Credentials verified against MongoDB
3. **Token Generation**: JWT access and refresh tokens created
4. **Session Creation**: Session stored in database
5. **Main Access**: Access granted to RAG PDF interface
6. **Token Validation**: Each request validates JWT token
7. **Logout**: Session invalidated, tokens cleared

## Security Considerations

### Production Deployment

1. **Change Default Passwords**
   ```bash
   export ADMIN_PASSWORD="your-secure-password"
   export JWT_SECRET="your-super-secret-jwt-key-min-32-chars"
   ```

2. **Use HTTPS**: Always deploy with SSL/TLS
3. **Environment Variables**: Never commit secrets to version control
4. **MongoDB Security**:
   - Use MongoDB Atlas with network access controls
   - Enable authentication on MongoDB
   - Use strong database credentials

### Rate Limiting

- Admin users bypass rate limits
- Default limit: 100 actions per hour
- Configurable per action type

### Activity Logging

- All login attempts are logged
- Failed logins tracked for security monitoring
- IP addresses and user agents recorded

## API Reference

### AuthManager Class

#### Methods:
- `connect()` - Connect to MongoDB
- `authenticate_user(username, password)` - Validate credentials
- `generate_tokens(user_id)` - Create JWT tokens
- `validate_token(token)` - Verify JWT token
- `create_session(user_id, tokens, ip_address)` - Store session
- `logout_user(token)` - Invalidate session
- `check_rate_limit(user_id, action, limit_per_hour)` - Rate limiting
- `log_activity(user_id, action, details)` - Activity logging
- `create_default_admin()` - Create admin account

### Login Page Functions

- `login_user(username, password)` - Handle login
- `logout_user()` - Handle logout
- `check_session()` - Verify active session
- `get_current_user_info()` - Get authenticated user details

## Deployment Notes

### Railway Deployment

1. Set environment variables in Railway dashboard
2. MongoDB Atlas recommended for cloud database
3. Ensure JWT_SECRET is set to a secure random value
4. Update ADMIN_PASSWORD for production

### Docker Deployment

```dockerfile
# Already included in Railway.Dockerfile
# Authentication dependencies automatically installed
```

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check MONGODB_URI format
   - Verify MongoDB service is running
   - Check network connectivity

2. **Authentication Failed**
   - Verify default admin account created
   - Check password in environment variables
   - Review MongoDB user collection

3. **JWT Token Errors**
   - Ensure JWT_SECRET is set
   - Check token expiration times
   - Verify token format in requests

### Logging

Authentication events are logged with levels:
- `INFO`: Successful operations
- `WARNING`: Non-critical issues
- `ERROR`: Failed operations

## Future Enhancements

Potential improvements:
- Password complexity requirements
- Multi-factor authentication
- User registration (currently disabled)
- Password reset functionality
- OAuth integration (Google, GitHub)
- Advanced rate limiting algorithms
- Session timeout configuration
- Audit log viewing interface

## Support

For issues related to the authentication system:
1. Check this documentation
2. Review log outputs
3. Verify environment configuration
4. Test with `test_auth.py` script