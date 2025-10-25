**Filename:** `README.md`

**Content:**

```markdown

# Rate Me Backend API

FastAPI backend for the Rate Me mobile application.

## Features

- User authentication (JWT)

- User ratings and reviews

- Competitions system

- QR code generation

- Profile management

## Tech Stack

- FastAPI (Python)

- MongoDB

- Motor (async MongoDB driver)

- JWT Authentication

## Deployment

### Railway Deployment

1. Fork this repository

2. Connect to Railway

3. Add environment variables:

- `MONGO_URL`: Your MongoDB connection string

- `SECRET_KEY`: JWT secret key

4. Deploy!

### Environment Variables Required

```

MONGO_URL=mongodb://...

SECRET_KEY=your-secret-key-here

```

## Local Development

```bash

pip install -r requirements.txt

uvicorn server:app --reload --port 8001

```

## API Documentation

Once deployed, visit `/docs` for Swagger documentation.

```
