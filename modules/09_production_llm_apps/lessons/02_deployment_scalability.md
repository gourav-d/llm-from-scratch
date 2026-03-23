# Lesson 2: Deployment & Scalability

**Learn to deploy and scale LLM applications from prototype to enterprise**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Containerization with Docker](#containerization-with-docker)
3. [Database Design](#database-design)
4. [Caching with Redis](#caching-with-redis)
5. [Cloud Deployment](#cloud-deployment)
6. [Kubernetes Orchestration](#kubernetes-orchestration)
7. [Load Balancing & Auto-Scaling](#load-balancing--auto-scaling)
8. [Performance Optimization](#performance-optimization)
9. [Complete Example](#complete-example)
10. [Exercises](#exercises)

---

## Introduction

### What You'll Learn

By the end of this lesson, you will:
- Containerize applications with Docker
- Deploy to cloud platforms (AWS/Azure/GCP)
- Use Kubernetes for orchestration
- Implement caching strategies
- Design scalable databases
- Handle 1000+ requests/minute

### Time Required
**10-12 hours** (including hands-on practice)

### Prerequisites
- Completed Lesson 1 (API Design)
- Basic Docker knowledge (helpful but not required)
- Cloud account (AWS/Azure/GCP free tier)

---

## Containerization with Docker

### Why Containers?

**Without Docker:**
```
"It works on my machine!" ❌
- Different Python versions
- Missing dependencies
- OS-specific issues
- Manual server setup
- Difficult to replicate
```

**With Docker:**
```
"It works everywhere!" ✅
- Consistent environment
- All dependencies bundled
- Works on any OS
- Easy deployment
- Reproducible builds
```

**C# Comparison:**

Docker works the same way for .NET applications:

```dockerfile
# .NET Dockerfile
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /app
COPY *.csproj ./
RUN dotnet restore
COPY . ./
RUN dotnet publish -c Release -o out

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/out .
ENTRYPOINT ["dotnet", "MyApp.dll"]
```

```dockerfile
# Python FastAPI Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Creating a Dockerfile

**Basic Dockerfile:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Production-optimized Dockerfile:**

```dockerfile
# Dockerfile.production
# Multi-stage build for smaller image size

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Add .local/bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Key improvements:**
- ✅ Multi-stage build (smaller image)
- ✅ Non-root user (security)
- ✅ Health check (monitoring)
- ✅ Multiple workers (performance)
- ✅ Optimized layers (faster builds)

---

### Docker Compose for Local Development

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  # Main API service
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/llm_app
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./:/app
    restart: unless-stopped

  # PostgreSQL database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=llm_app
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

**Running with Docker Compose:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

---

## Database Design

### Schema for LLM Applications

**PostgreSQL schema:**

```sql
-- schema.sql

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    quota_limit INTEGER DEFAULT 10000,
    quota_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens INTEGER,
    cost DECIMAL(10, 6),
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage tracking table
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(100),
    tokens INTEGER,
    cost DECIMAL(10, 6),
    duration_ms INTEGER,
    status_code INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_usage_logs_user_id ON usage_logs(user_id);
CREATE INDEX idx_usage_logs_created_at ON usage_logs(created_at);
```

**SQLAlchemy models (Python):**

```python
# models.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    quota_limit = Column(Integer, default=10000)
    quota_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    usage_logs = relationship("UsageLog", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tokens = Column(Integer)
    cost = Column(Float)
    model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    endpoint = Column(String(100))
    tokens = Column(Integer)
    cost = Column(Float)
    duration_ms = Column(Integer)
    status_code = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_logs")
```

**Database connection:**

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/llm_app")

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Using in FastAPI:**

```python
# main.py
from fastapi import Depends
from sqlalchemy.orm import Session
from database import get_db
from models import User, Conversation, Message

@app.post("/conversation")
async def create_conversation(user_id: str, db: Session = Depends(get_db)):
    """Create a new conversation"""
    conversation = Conversation(user_id=user_id, title="New Chat")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

@app.post("/conversation/{conversation_id}/message")
async def add_message(
    conversation_id: str,
    content: str,
    role: str,
    db: Session = Depends(get_db)
):
    """Add message to conversation"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tokens=count_tokens(content),
        cost=calculate_cost(content)
    )
    db.add(message)
    db.commit()
    return message
```

---

## Caching with Redis

### Why Caching?

**Without cache:**
- Every request hits LLM API ($$$)
- Slow responses (API latency)
- High costs for repeated queries

**With cache:**
- Common queries cached (free!)
- Instant responses (<10ms)
- 50-80% cost reduction

### Redis Implementation

```python
# cache.py
import redis
import json
import hashlib
from typing import Optional
import os

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

def get_cache_key(prompt: str, temperature: float, model: str) -> str:
    """Generate cache key from request parameters"""
    key_data = f"{prompt}:{temperature}:{model}"
    return hashlib.md5(key_data.encode()).hexdigest()

async def get_cached_response(prompt: str, temperature: float, model: str) -> Optional[str]:
    """Get cached response if exists"""
    cache_key = get_cache_key(prompt, temperature, model)

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        print(f"Cache read error: {e}")
        return None

async def set_cached_response(
    prompt: str,
    temperature: float,
    model: str,
    response: str,
    ttl: int = 3600  # 1 hour
):
    """Cache response with TTL"""
    cache_key = get_cache_key(prompt, temperature, model)

    try:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(response)
        )
    except Exception as e:
        print(f"Cache write error: {e}")

# Usage in endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    # Check cache first
    cached = await get_cached_response(
        request.message,
        request.temperature,
        "gpt-4"
    )

    if cached:
        return {"response": cached, "from_cache": True}

    # Call LLM if not cached
    response = await call_llm(request)

    # Cache the response
    await set_cached_response(
        request.message,
        request.temperature,
        "gpt-4",
        response
    )

    return {"response": response, "from_cache": False}
```

**Advanced caching strategies:**

```python
# Advanced cache patterns
class CacheStrategy:
    @staticmethod
    async def cache_with_sliding_expiration(key: str, value: str, ttl: int = 3600):
        """Reset TTL on each access"""
        redis_client.set(key, value)
        redis_client.expire(key, ttl)

    @staticmethod
    async def cache_popular_queries():
        """Pre-warm cache with popular queries"""
        popular = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain transformers"
        ]
        for query in popular:
            if not await get_cached_response(query, 0.7, "gpt-4"):
                response = await call_llm(query)
                await set_cached_response(query, 0.7, "gpt-4", response)

    @staticmethod
    async def invalidate_user_cache(user_id: str):
        """Clear all cache for a user"""
        pattern = f"user:{user_id}:*"
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
```

---

## Cloud Deployment

### AWS Deployment

**Option 1: AWS EC2 (Virtual Machine)**

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i mykey.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. Clone your repo
git clone https://github.com/yourusername/llm-api.git
cd llm-api

# 5. Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
EOF

# 6. Run with Docker Compose
docker-compose up -d

# 7. Configure nginx
sudo apt install nginx
sudo nano /etc/nginx/sites-available/api

# Add configuration:
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

sudo ln -s /etc/nginx/sites-available/api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 8. Setup SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

**Option 2: AWS ECS (Container Service)**

```yaml
# task-definition.json
{
  "family": "llm-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-ecr-repo/llm-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://..."
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:..."
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Azure Deployment

**Azure App Service:**

```bash
# 1. Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 2. Login
az login

# 3. Create resource group
az group create --name llm-api-rg --location eastus

# 4. Create App Service plan
az appservice plan create \
  --name llm-api-plan \
  --resource-group llm-api-rg \
  --sku B1 \
  --is-linux

# 5. Create web app
az webapp create \
  --resource-group llm-api-rg \
  --plan llm-api-plan \
  --name llm-api-app \
  --runtime "PYTHON:3.11"

# 6. Configure environment variables
az webapp config appsettings set \
  --resource-group llm-api-rg \
  --name llm-api-app \
  --settings OPENAI_API_KEY="sk-..."

# 7. Deploy code
az webapp up \
  --resource-group llm-api-rg \
  --name llm-api-app \
  --runtime "PYTHON:3.11"
```

### GCP Deployment

**Cloud Run (Serverless):**

```bash
# 1. Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# 2. Initialize
gcloud init

# 3. Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-api

# 4. Deploy to Cloud Run
gcloud run deploy llm-api \
  --image gcr.io/PROJECT_ID/llm-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-... \
  --memory 1Gi \
  --cpu 2
```

---

## Kubernetes Orchestration

### Why Kubernetes?

**Benefits:**
- Auto-scaling based on load
- Self-healing (auto-restart failed containers)
- Load balancing
- Rolling updates with zero downtime
- Works across all clouds

**Kubernetes manifests:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: api
        image: your-registry/llm-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Deploy to Kubernetes:**

```bash
# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get pods
kubectl get services
kubectl get hpa

# View logs
kubectl logs -f deployment/llm-api

# Scale manually
kubectl scale deployment llm-api --replicas=5
```

---

## Load Balancing & Auto-Scaling

### Application Load Balancer

```nginx
# nginx.conf
upstream api_servers {
    least_conn;  # Use least connections algorithm
    server api1:8000 weight=1;
    server api2:8000 weight=1;
    server api3:8000 weight=1;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
}
```

### Auto-scaling Configuration

**AWS Auto Scaling Group:**

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name llm-api-template \
  --version-description "v1" \
  --launch-template-data '{
    "ImageId": "ami-xxxxx",
    "InstanceType": "t3.medium",
    "UserData": "base64-encoded-startup-script"
  }'

# Create auto scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name llm-api-asg \
  --launch-template LaunchTemplateName=llm-api-template \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3 \
  --target-group-arns arn:aws:elasticloadbalancing:...

# Create scaling policy
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name llm-api-asg \
  --policy-name scale-on-cpu \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    },
    "TargetValue": 70.0
  }'
```

---

## Performance Optimization

### Optimization Checklist

```python
# 1. Use async/await everywhere
@app.post("/chat")
async def chat(request: ChatRequest):
    response = await async_llm_call(request)  # Non-blocking
    return response

# 2. Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)

# 3. Batch database operations
messages = [Message(content=f"Message {i}") for i in range(100)]
db.bulk_save_objects(messages)
db.commit()  # One commit for all

# 4. Use indexes
# Already created in schema.sql

# 5. Enable gzip compression
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 6. Cache static responses
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")

@app.get("/models")
@cache(expire=3600)  # Cache for 1 hour
async def get_models():
    return {"models": ["gpt-4", "gpt-3.5-turbo"]}
```

---

## Complete Example

**Complete production deployment:**

```bash
# Directory structure
llm-api/
├── Dockerfile
├── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── main.py
├── models.py
├── database.py
├── cache.py
└── requirements.txt

# Deploy to production
# 1. Build and push image
docker build -t your-registry/llm-api:v1.0 .
docker push your-registry/llm-api:v1.0

# 2. Deploy to Kubernetes
kubectl apply -f kubernetes/

# 3. Verify deployment
kubectl get pods
kubectl get svc

# 4. Test API
curl http://your-loadbalancer-ip/health
```

---

## Exercises

### Exercise 1: Dockerize Your API (2 hours)
1. Create Dockerfile
2. Build image
3. Run container locally
4. Test endpoints

### Exercise 2: Add Database (3 hours)
1. Set up PostgreSQL
2. Create schema
3. Implement models
4. Add CRUD endpoints

### Exercise 3: Deploy to Cloud (3 hours)
1. Choose cloud provider
2. Deploy application
3. Configure domain
4. Test production API

### Exercise 4: Add Caching (2 hours)
1. Set up Redis
2. Implement cache layer
3. Measure performance improvement
4. Monitor cache hit rate

---

## Summary

### What You Learned
- ✅ Docker containerization
- ✅ Database design for LLM apps
- ✅ Redis caching strategies
- ✅ Cloud deployment (AWS/Azure/GCP)
- ✅ Kubernetes orchestration
- ✅ Auto-scaling configuration

### Key Takeaways
1. **Containers are essential** - Use Docker everywhere
2. **Cache aggressively** - Save 50-80% on costs
3. **Design for scale** - Start with scalable architecture
4. **Monitor everything** - Know what's happening
5. **Automate deployment** - CI/CD from day one

### Next Lesson
**Lesson 3:** Monitoring & Observability
- Prometheus metrics
- Grafana dashboards
- Distributed tracing
- Log aggregation

---

**Time to complete:** 10-12 hours
**Difficulty:** Intermediate to Advanced
**Next lesson:** 03_monitoring_observability.md
