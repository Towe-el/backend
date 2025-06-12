from fastapi import APIRouter
import requests
import socket
import os
import asyncio
import subprocess
from typing import Dict, Any, List

router = APIRouter(
    prefix="/debug",
    tags=["debug"]
)

@router.get("/network-info")
async def get_network_info() -> Dict[str, Any]:
    """
    Get Cloud Run instance network information, including outbound IP address
    Used to determine the IP range that MongoDB Atlas whitelist should be configured for
    """
    result = {
        "environment": "unknown",
        "local_ip": None,
        "outbound_ip": None,
        "hostname": None,
        "vpc_info": {},
        "mongodb_test": "not_tested"
    }
    
    # 1. Detect environment
    if os.getenv("K_SERVICE"):
        result["environment"] = "cloud_run"
        result["service_name"] = os.getenv("K_SERVICE")
        result["revision"] = os.getenv("K_REVISION")
    elif os.getenv("GAE_APPLICATION"):
        result["environment"] = "app_engine"
    else:
        result["environment"] = "local_or_other"
    
    # 2. Get hostname
    try:
        result["hostname"] = socket.gethostname()
    except Exception as e:
        result["hostname"] = f"error: {e}"
    
    # 3. Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        result["local_ip"] = s.getsockname()[0]
        s.close()
    except Exception as e:
        result["local_ip"] = f"error: {e}"
    
    # 4. Get outbound public IP
    ip_services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip", 
        "https://icanhazip.com",
        "https://ipecho.net/plain"
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service, timeout=10)
            if response.status_code == 200:
                result["outbound_ip"] = response.text.strip()
                result["ip_service_used"] = service
                break
        except Exception as e:
            continue
    
    # 5. VPC related information (if in GCP environment)
    if result["environment"] == "cloud_run":
        try:
            # Try to get GCP metadata
            metadata_headers = {"Metadata-Flavor": "Google"}
            
            # Get project ID
            try:
                resp = requests.get(
                    "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                    headers=metadata_headers,
                    timeout=5
                )
                if resp.status_code == 200:
                    result["vpc_info"]["project_id"] = resp.text
            except:
                pass
            
            # Get region
            try:
                resp = requests.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/region",
                    headers=metadata_headers,
                    timeout=5
                )
                if resp.status_code == 200:
                    result["vpc_info"]["region"] = resp.text.split("/")[-1]
            except:
                pass
                
            # Get network interface information
            try:
                resp = requests.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip",
                    headers=metadata_headers,
                    timeout=5
                )
                if resp.status_code == 200:
                    result["vpc_info"]["internal_ip"] = resp.text
            except:
                pass
                
        except Exception as e:
            result["vpc_info"]["error"] = str(e)
    
    # 6. Test MongoDB connection (optional)
    try:
        from app.database import sync_health_check
        
        if sync_health_check():
            result["mongodb_test"] = "success"
        else:
            result["mongodb_test"] = "failed: health check failed"
    except Exception as e:
        result["mongodb_test"] = f"failed: {str(e)[:100]}"
    
    return result