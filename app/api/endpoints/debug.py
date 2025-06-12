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

@router.get("/network-diagnostics")
async def get_network_diagnostics() -> Dict[str, Any]:
    """
    详细的网络连接诊断，用于排查VPC配置问题
    """
    diagnostics = {
        "dns_tests": {},
        "connectivity_tests": {},
        "route_info": {},
        "interface_info": {},
        "recommendations": []
    }
    
    # DNS解析测试
    test_domains = [
        "google.com",
        "api.ipify.org", 
        "mongodb.net",
        "ac-df8rkil-shard-00-00.04rdzbd.mongodb.net"
    ]
    
    for domain in test_domains:
        try:
            ip = socket.gethostbyname(domain)
            diagnostics["dns_tests"][domain] = {"status": "success", "ip": ip}
        except Exception as e:
            diagnostics["dns_tests"][domain] = {"status": "failed", "error": str(e)}
    
    # 连接测试
    test_connections = [
        {"host": "8.8.8.8", "port": 53, "name": "Google DNS"},
        {"host": "api.ipify.org", "port": 443, "name": "IP检测服务"},
        {"host": "ac-df8rkil-shard-00-00.04rdzbd.mongodb.net", "port": 27017, "name": "MongoDB Atlas"}
    ]
    
    for test in test_connections:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((test["host"], test["port"]))
            sock.close()
            
            if result == 0:
                diagnostics["connectivity_tests"][test["name"]] = {
                    "status": "success", 
                    "host": test["host"], 
                    "port": test["port"]
                }
            else:
                diagnostics["connectivity_tests"][test["name"]] = {
                    "status": "failed", 
                    "host": test["host"], 
                    "port": test["port"],
                    "error": f"Connection failed with code {result}"
                }
        except Exception as e:
            diagnostics["connectivity_tests"][test["name"]] = {
                "status": "error", 
                "host": test["host"], 
                "port": test["port"],
                "error": str(e)
            }
    
    # 网络接口信息
    try:
        import netifaces
        interfaces = netifaces.interfaces()
        diagnostics["interface_info"]["available_interfaces"] = interfaces
        
        for interface in interfaces:
            try:
                addrs = netifaces.ifaddresses(interface)
                diagnostics["interface_info"][interface] = addrs
            except:
                pass
    except ImportError:
        diagnostics["interface_info"]["error"] = "netifaces module not available"
    except Exception as e:
        diagnostics["interface_info"]["error"] = str(e)
    
    # 环境变量检查
    vpc_env_vars = [
        "VPC_CONNECTOR_NAME",
        "VPC_EGRESS_SETTING", 
        "GOOGLE_CLOUD_PROJECT",
        "K_SERVICE",
        "K_REVISION"
    ]
    
    diagnostics["environment_vars"] = {}
    for var in vpc_env_vars:
        diagnostics["environment_vars"][var] = os.getenv(var, "not_set")
    
    # 基于诊断结果提供建议
    recommendations = []
    
    # DNS问题检查
    dns_failures = [k for k, v in diagnostics["dns_tests"].items() if v["status"] == "failed"]
    if dns_failures:
        recommendations.append(f"DNS解析失败: {', '.join(dns_failures)}。可能是VPC DNS配置问题。")
    
    # 连接问题检查
    conn_failures = [k for k, v in diagnostics["connectivity_tests"].items() if v["status"] in ["failed", "error"]]
    if conn_failures:
        recommendations.append(f"连接测试失败: {', '.join(conn_failures)}。可能是防火墙或路由问题。")
    
    # 如果所有外部连接都失败
    if len(conn_failures) == len(test_connections):
        recommendations.extend([
            "所有外部连接都失败，建议检查:",
            "1. VPC连接器的egress设置是否正确",
            "2. VPC网络的路由表配置",
            "3. 防火墙规则是否阻止了出站流量",
            "4. Cloud NAT是否正确配置（如果使用private ranges only）"
        ])
    
    diagnostics["recommendations"] = recommendations
    
    return diagnostics

@router.get("/vpc-troubleshooting")
async def get_vpc_troubleshooting() -> Dict[str, Any]:
    """
    VPC连接器问题排查和修复建议
    """
    troubleshooting = {
        "current_config_check": {},
        "common_issues": [],
        "fix_commands": [],
        "step_by_step_guide": []
    }
    
    # 当前配置检查
    network_info = await get_network_info()
    
    troubleshooting["current_config_check"] = {
        "can_detect_outbound_ip": network_info.get("outbound_ip") is not None,
        "mongodb_connection": "success" in network_info.get("mongodb_test", ""),
        "environment": network_info.get("environment"),
        "local_ip": network_info.get("local_ip")
    }
    
    # 常见问题和解决方案
    if not troubleshooting["current_config_check"]["can_detect_outbound_ip"]:
        troubleshooting["common_issues"].extend([
            "无法检测出站IP地址，可能原因:",
            "- VPC连接器的egress设置为all-traffic但没有配置Cloud NAT",
            "- VPC网络路由配置错误",
            "- 防火墙规则阻止了HTTPS出站流量"
        ])
        
        troubleshooting["fix_commands"].extend([
            "# 检查VPC连接器状态",
            "gcloud compute networks vpc-access connectors describe CONNECTOR_NAME --region=europe-west1",
            "",
            "# 检查是否需要Cloud NAT (如果使用all-traffic egress)",
            "gcloud compute routers nats list --router=ROUTER_NAME --region=europe-west1",
            "",
            "# 临时解决方案：改回private-ranges-only + 配置Cloud NAT",
            "gcloud compute networks vpc-access connectors update CONNECTOR_NAME \\",
            "    --region=europe-west1 \\",
            "    --egress=private-ranges-only"
        ])
    
    # 分步骤修复指南
    troubleshooting["step_by_step_guide"] = [
        "1. 确认VPC连接器状态和配置",
        "2. 如果使用all-traffic egress，确保配置了Cloud NAT",
        "3. 如果不需要所有流量走VPC，改回private-ranges-only",
        "4. 检查防火墙规则允许必要的出站流量",
        "5. 重新部署Cloud Run服务以应用配置变更",
        "6. 使用 /debug/network-info 验证修复结果"
    ]
    
    return troubleshooting

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