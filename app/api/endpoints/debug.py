from fastapi import APIRouter
import requests
import socket
import os
from typing import Dict, Any

router = APIRouter(
    prefix="/debug",
    tags=["debug"]
)

@router.get("/network-info")
async def get_network_info() -> Dict[str, Any]:
    """
    获取Cloud Run实例的网络信息，包括出站IP地址
    用于确定MongoDB Atlas白名单应该配置的IP范围
    """
    result = {
        "environment": "unknown",
        "local_ip": None,
        "outbound_ip": None,
        "hostname": None,
        "vpc_info": {},
        "mongodb_test": "not_tested"
    }
    
    # 1. 检测环境
    if os.getenv("K_SERVICE"):
        result["environment"] = "cloud_run"
        result["service_name"] = os.getenv("K_SERVICE")
        result["revision"] = os.getenv("K_REVISION")
    elif os.getenv("GAE_APPLICATION"):
        result["environment"] = "app_engine"
    else:
        result["environment"] = "local_or_other"
    
    # 2. 获取主机名
    try:
        result["hostname"] = socket.gethostname()
    except Exception as e:
        result["hostname"] = f"error: {e}"
    
    # 3. 获取本地IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        result["local_ip"] = s.getsockname()[0]
        s.close()
    except Exception as e:
        result["local_ip"] = f"error: {e}"
    
    # 4. 获取出站公网IP
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
    
    # 5. VPC相关信息（如果在GCP环境中）
    if result["environment"] == "cloud_run":
        try:
            # 尝试获取GCP元数据
            metadata_headers = {"Metadata-Flavor": "Google"}
            
            # 获取项目ID
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
            
            # 获取区域
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
                
            # 获取网络接口信息
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
    
    # 6. 测试MongoDB连接（可选）
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        mongo_uri = os.getenv("MONGODB_URI")
        if mongo_uri:
            client = MongoClient(
                mongo_uri,
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=3000
            )
            client.admin.command('ping')
            client.close()
            result["mongodb_test"] = "success"
        else:
            result["mongodb_test"] = "no_uri"
    except Exception as e:
        result["mongodb_test"] = f"failed: {str(e)[:100]}"
    
    return result

@router.get("/ip-recommendations")
async def get_ip_recommendations() -> Dict[str, Any]:
    """
    基于当前环境提供MongoDB Atlas IP白名单配置建议
    """
    network_info = await get_network_info()
    
    recommendations = {
        "current_outbound_ip": network_info.get("outbound_ip"),
        "environment": network_info.get("environment"),
        "mongodb_status": network_info.get("mongodb_test"),
        "recommendations": []
    }
    
    outbound_ip = network_info.get("outbound_ip")
    
    if outbound_ip:
        # 基本IP配置
        recommendations["recommendations"].append({
            "type": "exact_ip",
            "value": f"{outbound_ip}/32",
            "description": "当前实例的确切出站IP地址"
        })
        
        # 如果是Cloud Run环境，提供VPC范围建议
        if network_info.get("environment") == "cloud_run":
            ip_parts = outbound_ip.split('.')
            if len(ip_parts) == 4:
                # 常见的GCP VPC IP范围
                vpc_ranges = [
                    f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24",
                    f"{ip_parts[0]}.{ip_parts[1]}.0.0/16",
                    "10.0.0.0/8",      # 私有网络范围A
                    "172.16.0.0/12",   # 私有网络范围B  
                    "192.168.0.0/16"   # 私有网络范围C
                ]
                
                for vpc_range in vpc_ranges:
                    recommendations["recommendations"].append({
                        "type": "vpc_range",
                        "value": vpc_range,
                        "description": f"可能的VPC IP范围（基于 {outbound_ip}）"
                    })
        
        # 安全建议
        recommendations["security_notes"] = [
            "生产环境中避免使用 0.0.0.0/0",
            "定期检查和更新IP白名单",
            "考虑使用VPC私有连接以提高安全性"
        ]
        
        # 操作步骤
        recommendations["setup_steps"] = [
            "1. 登录MongoDB Atlas控制台",
            "2. 进入 Network Access",
            "3. 点击 'ADD IP ADDRESS'",
            f"4. 添加IP: {outbound_ip}/32",
            "5. 如果使用VPC连接器，考虑添加整个VPC范围",
            "6. 保存并等待配置生效（通常1-2分钟）"
        ]
    else:
        recommendations["error"] = "无法检测出站IP地址"
    
    return recommendations 