# 🚀 Deployment Guide - NASA Exoplanet Detection System

This guide covers various deployment options for the NASA Exoplanet Detection System, from local development to cloud production environments.

## 📋 Quick Start Options

### Option 1: Automated Setup (Recommended)
```bash
# Windows
start.bat

# Or cross-platform
python run.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Train models (10-15 minutes)
python train_models.py

# Start web server
python app.py
```

### Option 3: Demo Mode (No Training Required)
```bash
python demo.py
```

## 🖥️ Local Development

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for data and models
- **Network**: Internet connection for NASA data download

### Development Setup
```bash
# Clone/download project
cd NASA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python app.py
```

### Development Features
- **Hot Reload**: Flask debug mode for live code changes
- **Detailed Logging**: Console output for debugging
- **Model Retraining**: Easy to retrain models with new data
- **Custom Data**: Add your own light curve datasets

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Prepare Heroku Files**
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Create runtime.txt
echo "python-3.9.18" > runtime.txt

# Install Heroku CLI and login
heroku login
```

2. **Deploy to Heroku**
```bash
# Create Heroku app
heroku create nasa-exoplanet-detector

# Set environment variables
heroku config:set FLASK_ENV=production

# Deploy
git init
git add .
git commit -m "Initial deployment"
git push heroku main

# Scale web dyno
heroku ps:scale web=1
```

3. **Pre-trained Models**
Since Heroku has limited build time, consider:
- Pre-training models locally
- Uploading to cloud storage (AWS S3, Google Cloud Storage)
- Loading models from cloud storage on startup

### AWS Deployment

#### EC2 Instance
```bash
# Launch Ubuntu EC2 instance (t3.medium recommended)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Python and dependencies
sudo apt update
sudo apt install python3 python3-pip nginx

# Clone project
git clone your-repo-url
cd NASA

# Install dependencies
pip3 install -r requirements.txt

# Train models (or download pre-trained)
python3 train_models.py

# Install Gunicorn
pip3 install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init nasa-exoplanet-detector

# Create environment
eb create production

# Deploy
eb deploy
```

### Google Cloud Platform

#### App Engine
```yaml
# Create app.yaml
runtime: python39

env_variables:
  FLASK_ENV: production

automatic_scaling:
  min_instances: 1
  max_instances: 10

resources:
  cpu: 2
  memory_gb: 4
```

```bash
# Deploy to App Engine
gcloud app deploy
```

#### Cloud Run
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Pre-train models (optional)
# RUN python train_models.py

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
```

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/nasa-exoplanet
gcloud run deploy --image gcr.io/PROJECT_ID/nasa-exoplanet --platform managed
```

### Azure Deployment

#### App Service
```bash
# Install Azure CLI
az login

# Create resource group
az group create --name nasa-exoplanet-rg --location eastus

# Create App Service plan
az appservice plan create --name nasa-plan --resource-group nasa-exoplanet-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group nasa-exoplanet-rg --plan nasa-plan --name nasa-exoplanet-detector --runtime "PYTHON|3.9"

# Deploy code
az webapp deployment source config-zip --resource-group nasa-exoplanet-rg --name nasa-exoplanet-detector --src deployment.zip
```

## 🐳 Docker Deployment

### Basic Docker Setup
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data models plots

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Docker Compose (with Redis for caching)
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
```

### Build and Run
```bash
# Build Docker image
docker build -t nasa-exoplanet-detector .

# Run container
docker run -p 5000:5000 nasa-exoplanet-detector

# Or use Docker Compose
docker-compose up -d
```

## 🔧 Production Optimizations

### Performance Tuning

#### Model Optimization
```python
# In models.py - add model quantization
import tensorflow as tf

def optimize_model(model):
    """Optimize model for production"""
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    return tflite_model

# Use optimized models in production
```

#### Caching Strategy
```python
# In app.py - add Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

@app.route('/api/predict', methods=['POST'])
@cache_result(expiration=1800)  # Cache for 30 minutes
def predict_exoplanet():
    # ... existing code
```

### Security Hardening

#### Environment Variables
```bash
# Create .env file
FLASK_SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url
REDIS_URL=your-redis-url
NASA_API_KEY=your-nasa-api-key
```

#### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_exoplanet():
    # ... existing code
```

### Monitoring and Logging

#### Application Monitoring
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/nasa_exoplanet.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

#### Health Check Endpoint
```python
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check if models are loaded
        if not ensemble or not ensemble.models:
            return jsonify({'status': 'unhealthy', 'reason': 'models not loaded'}), 503
        
        # Check database connection (if applicable)
        # ... database check code
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'models_loaded': len(ensemble.models)
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503
```

## 📊 Scaling Considerations

### Horizontal Scaling
- **Load Balancer**: Use Nginx or cloud load balancers
- **Multiple Instances**: Run multiple app instances behind load balancer
- **Database**: Use PostgreSQL or MongoDB for persistent data
- **File Storage**: Use cloud storage (S3, GCS) for model files

### Vertical Scaling
- **CPU**: Multi-core instances for parallel model inference
- **Memory**: 8GB+ RAM for large model ensembles
- **GPU**: Optional GPU instances for faster deep learning inference

### Auto-scaling Configuration
```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nasa-exoplanet-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nasa-exoplanet-detector
  template:
    metadata:
      labels:
        app: nasa-exoplanet-detector
    spec:
      containers:
      - name: app
        image: nasa-exoplanet-detector:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nasa-exoplanet-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nasa-exoplanet-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🔒 Security Best Practices

### HTTPS Configuration
```nginx
# Nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Input Validation
```python
from marshmallow import Schema, fields, validate

class LightCurveSchema(Schema):
    time = fields.List(fields.Float(), required=True, validate=validate.Length(min=100, max=10000))
    flux = fields.List(fields.Float(), required=True, validate=validate.Length(min=100, max=10000))

@app.route('/api/predict', methods=['POST'])
def predict_exoplanet():
    schema = LightCurveSchema()
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify({'error': 'Invalid input', 'details': err.messages}), 400
    
    # ... rest of the function
```

## 📈 Monitoring and Analytics

### Application Metrics
```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')

# Custom metrics
prediction_counter = metrics.counter(
    'predictions_total', 'Total predictions made',
    labels={'model': lambda: request.view_args['model'] if request.view_args else 'unknown'}
)

@app.route('/api/predict', methods=['POST'])
@prediction_counter
def predict_exoplanet():
    # ... existing code
```

### Error Tracking
```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## 🚀 Deployment Checklist

### Pre-deployment
- [ ] All dependencies installed and tested
- [ ] Models trained and validated
- [ ] Environment variables configured
- [ ] Security measures implemented
- [ ] Performance optimizations applied
- [ ] Monitoring and logging configured

### Deployment
- [ ] Application deployed to target environment
- [ ] Health checks passing
- [ ] SSL certificate configured (for production)
- [ ] Load balancer configured (if applicable)
- [ ] Auto-scaling rules set up
- [ ] Backup and recovery procedures in place

### Post-deployment
- [ ] Application accessible and functional
- [ ] All API endpoints responding correctly
- [ ] Monitoring dashboards configured
- [ ] Error tracking active
- [ ] Performance metrics baseline established
- [ ] Documentation updated

## 🆘 Troubleshooting

### Common Issues

#### Models Not Loading
```bash
# Check if model files exist
ls -la models/

# Retrain models if missing
python train_models.py

# Check file permissions
chmod 644 models/*
```

#### Memory Issues
```python
# Reduce model complexity in models.py
# Use model quantization
# Implement model lazy loading
```

#### Slow Performance
```python
# Enable model caching
# Use Redis for session storage
# Optimize database queries
# Implement CDN for static assets
```

#### Connection Issues
```bash
# Check firewall settings
sudo ufw status

# Check port availability
netstat -tulpn | grep :5000

# Test local connection
curl http://localhost:5000/health
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review application logs
3. Verify system requirements
4. Test with the demo script first
5. Create an issue with detailed error information

---

**Happy Deploying! 🚀 May your exoplanet detection system discover new worlds!** 🌍✨
