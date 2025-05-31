# Toweel Backend API

## Environment Variables

Before running the project, set the following environment variables first：

Create a `.env` file in the `backend\` directory. Copy paste the information transferred privately in the chat group.

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