# Toweel Backend API

## Environment Variables

Before running the project, set the following environment variables first：

1. Create a `credentials` directory in the project root:
```bash
mkdir credentials
```

2. Place your Google Cloud service account key file in the credentials directory as `service-account.json`

3. Create a `.env` file in the project root with the following content:
```
MONGODB_URL=MongoDB connection string
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Quick Start

1. Make sure you have completed the environment setup above
2. Run Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. API service enables at: http://localhost:8000 
4. Visualize the API service at: http://localhost:8000/docs 

## Folder Structure

### DataProcess/
Dataset and scripts to process data before uploading to MongoDB.

These files are independent to the backend services. All of them are tested locally.