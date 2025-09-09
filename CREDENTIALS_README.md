# HFT Neurosymbolic AI System - Credentials Management

This document explains how to manage credentials and secrets for the HFT Neurosymbolic AI System.

## 🔐 Credential Files

### `credentials.env` (ACTUAL CREDENTIALS - NEVER COMMIT)
- Contains real usernames and passwords for all platforms
- **CRITICAL**: This file is in `.gitignore` and should NEVER be committed to version control
- Contains sensitive information that must be kept secure

### `credentials.env.template` (TEMPLATE - SAFE TO COMMIT)
- Template file with placeholder values
- Safe to commit to version control
- Used as a reference for setting up credentials

## 🚀 Quick Setup

1. **Copy the template:**
   ```bash
   cp credentials.env.template credentials.env
   ```

2. **Edit with your credentials:**
   ```bash
   nano credentials.env  # or use your preferred editor
   ```

3. **Replace all placeholder values** with your actual credentials

## 📋 Platform Credentials Included

### Core Database Systems
- **PostgreSQL**: Main database for Hasura GraphQL Engine
- **Neo4j**: Graph database for relationship data
- **Dgraph**: Distributed graph database
- **Apache Jena Fuseki**: SPARQL endpoint for RDF data

### Caching & Session Management
- **Redis**: In-memory cache and session store

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization

### Application Services
- **FastAPI**: Main API server
- **Streamlit**: Interactive dashboard

### External APIs (Optional)
- **Yahoo Finance**: Market data
- **CCXT Exchanges**: Cryptocurrency trading
- **Alternative Data Providers**: Alpha Vantage, Quandl, Polygon

### Security & Authentication
- **JWT**: Token-based authentication
- **Encryption**: Data encryption keys

### Cloud & Deployment
- **AWS**: Cloud services
- **Google Cloud**: Cloud services
- **Azure**: Cloud services

### Trading Platforms (Optional)
- **Interactive Brokers**: Professional trading
- **TD Ameritrade**: Retail trading

### AI/ML Services (Optional)
- **OpenAI**: GPT models
- **Hugging Face**: Pre-trained models
- **Weights & Biases**: Experiment tracking
- **MLflow**: Model registry

## 🔒 Security Best Practices

### Password Requirements
- Use strong, unique passwords for each service
- Minimum 16 characters with mixed case, numbers, and symbols
- Avoid dictionary words or common patterns
- Consider using a password manager

### Password Rotation
- Rotate passwords every 90 days
- Update credentials immediately if compromised
- Use different passwords for different environments (dev/staging/prod)

### Environment Separation
Create separate credential files for different environments:
```bash
credentials.dev.env      # Development environment
credentials.staging.env  # Staging environment
credentials.prod.env     # Production environment
```

## 🐳 Docker Integration

### Using Environment Files
```bash
# Load credentials into Docker containers
docker-compose --env-file credentials.env up -d
```

### Docker Secrets (Recommended for Production)
```yaml
# docker-compose.yml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

## ☸️ Kubernetes Integration

### Using Kubernetes Secrets
```bash
# Create secret from environment file
kubectl create secret generic hft-credentials --from-env-file=credentials.env

# Use in deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hft-app
spec:
  template:
    spec:
      containers:
      - name: app
        envFrom:
        - secretRef:
            name: hft-credentials
```

## 🏢 Enterprise Credential Management

### HashiCorp Vault Integration
For enterprise deployments, consider using HashiCorp Vault:

```bash
# Store credentials in Vault
vault kv put secret/hft/credentials \
  postgres_password="your_password" \
  redis_password="your_password"

# Retrieve in application
vault kv get -field=postgres_password secret/hft/credentials
```

## 🔍 Credential Validation

### Test Credentials
```bash
# Test database connection
python scripts/test_credentials.py

# Test all services
python scripts/validate_all_credentials.py
```

### Health Checks
The system includes health check endpoints that validate credentials:
- `GET /health` - Overall system health
- `GET /api/v1/health/detailed` - Detailed service health

## 🚨 Incident Response

### If Credentials Are Compromised
1. **Immediately rotate** all affected passwords
2. **Update** the credentials.env file
3. **Restart** all services with new credentials
4. **Audit** access logs for unauthorized usage
5. **Notify** relevant team members

### Backup & Recovery
- Keep encrypted backups of credential files
- Store backup credentials in secure, separate location
- Document recovery procedures

## 📞 Support

### Common Issues
- **Connection refused**: Check host/port and credentials
- **Authentication failed**: Verify username/password
- **Permission denied**: Check user permissions in database

### Getting Help
- Check application logs: `docker logs hft_neurosymbolic_app`
- Review service health: `curl http://localhost:8001/health`
- Consult platform-specific documentation

## 📝 Notes

- All default passwords should be changed in production
- Use environment-specific configurations
- Regularly audit credential access
- Consider implementing credential rotation automation
- Monitor for credential exposure in logs or error messages

---

**Remember**: Security is everyone's responsibility. Keep credentials secure and follow best practices!
