import json
import socket
import ssl
import requests
import whois
from datetime import datetime
import dns.resolver
import subprocess
import platform
import nmap
import concurrent.futures
import urllib3
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import socket
import struct
import sys
import os
from typing import Dict, Any

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def get_dns_records(domain: str) -> Dict[str, Any]:
    """Get various DNS records for the domain"""
    record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME', 'PTR']
    dns_info = {}

    for record_type in record_types:
        try:
            answers = dns.resolver.resolve(domain, record_type)
            dns_info[record_type] = [str(answer) for answer in answers]
        except Exception as e:
            dns_info[record_type] = f"Failed to retrieve {record_type} record: {str(e)}"

    return dns_info

def get_ssl_details(host: str, port: int) -> Dict[str, Any]:
    """Get detailed SSL/TLS certificate information"""
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port)) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssl_sock:
                cert = ssl_sock.getpeercert(binary_form=True)
                x509_cert = x509.load_der_x509_certificate(cert, default_backend())

                return {
                    "subject": str(x509_cert.subject),
                    "issuer": str(x509_cert.issuer),
                    "version": x509_cert.version,
                    "serial_number": x509_cert.serial_number,
                    "not_valid_before": x509_cert.not_valid_before,
                    "not_valid_after": x509_cert.not_valid_after,
                    "signature_algorithm": x509_cert.signature_algorithm_oid._name,
                    "public_key_type": type(x509_cert.public_key()).__name__,
                    "public_key_size": x509_cert.public_key().key_size,
                }
    except Exception as e:
        return {"error": f"Failed to get SSL details: {str(e)}"}

def perform_traceroute(host: str) -> list:
    """Perform traceroute and return results"""
    traceroute_data = []

    if platform.system().lower() == "windows":
        command = ["tracert", host]
    else:
        command = ["traceroute", "-n", host]

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        for line in output.split('\n'):
            if line.strip():
                traceroute_data.append(line)
        return traceroute_data
    except Exception as e:
        return [f"Traceroute failed: {str(e)}"]

def scan_ports(host: str, start_port: int, end_port: int) -> Dict[int, str]:
    """Scan a range of ports"""
    nm = nmap.PortScanner()
    try:
        nm.scan(host, f"{start_port}-{end_port}", arguments="-sT -sV -T4")
        return nm[host].all_tcp()
    except Exception as e:
        return {-1: f"Port scan failed: {str(e)}"}

def get_http_security_headers(url: str) -> Dict[str, str]:
    """Check for security-related HTTP headers"""
    security_headers = {
        'Strict-Transport-Security',
        'Content-Security-Policy',
        'X-Frame-Options',
        'X-Content-Type-Options',
        'X-XSS-Protection',
        'Referrer-Policy',
        'Feature-Policy',
        'Permissions-Policy'
    }

    try:
        response = requests.get(url, verify=False)
        return {header: response.headers.get(header, 'Not set')
                for header in security_headers}
    except Exception as e:
        return {"error": f"Failed to get security headers: {str(e)}"}

def check_server_status(host: str, port: int, timeout: float = 5.0):
    """
    Enhanced server status check with additional information
    """
    result = {
        "timestamp": datetime.now(),
        "host": host,
        "port": port,
        "basic_info": {
            "reachable": False,
            "port_open": False,
            "ip_address": None,
            "hostname": None,
            "error": None,
        },
        "dns_info": None,
        "ssl_details": None,
        "security_headers": None,
        "traceroute": None,
        "port_scan": None,
        "whois": None,
        "geolocation": None,
        "http_headers": None,
        "network_info": {},
        "server_fingerprint": {},
    }

    # Basic connectivity check
    try:
        result["basic_info"]["ip_address"] = socket.gethostbyname(host)
        result["basic_info"]["hostname"] = socket.getfqdn(host)
        result["basic_info"]["reachable"] = True

        with socket.create_connection((host, port), timeout) as sock:
            result["basic_info"]["port_open"] = True

    except Exception as e:
        result["basic_info"]["error"] = str(e)

    # DNS Information
    try:
        result["dns_info"] = get_dns_records(host)
    except Exception as e:
        result["dns_info"] = f"DNS lookup failed: {str(e)}"

    # SSL/TLS Details
    if port in [443, 993, 995, 8443]:
        result["ssl_details"] = get_ssl_details(host, port)

    # Security Headers
    try:
        url = f"https://{host}:{port}" if port == 443 else f"http://{host}:{port}"
        result["security_headers"] = get_http_security_headers(url)
    except Exception as e:
        result["security_headers"] = f"Failed to get security headers: {str(e)}"

    # Traceroute
    result["traceroute"] = perform_traceroute(host)

    # Port Scan (limited range)
    result["port_scan"] = scan_ports(host, port-5 if port > 5 else 1, port+5)

    # WHOIS Information
    try:
        domain_info = whois.whois(host)
        result["whois"] = {
            "registrar": domain_info.registrar,
            "creation_date": domain_info.creation_date,
            "expiration_date": domain_info.expiration_date,
            "registrant": domain_info.registrant,
            "admin_email": domain_info.admin_email,
            "status": domain_info.status,
        }
    except Exception as e:
        result["whois"] = f"WHOIS lookup failed: {str(e)}"

    # Geolocation
    try:
        response = requests.get(f"https://ipinfo.io/{result['basic_info']['ip_address']}/json")
        if response.status_code == 200:
            result["geolocation"] = response.json()
    except Exception as e:
        result["geolocation"] = f"Geolocation lookup failed: {str(e)}"

    # Additional Network Information
    try:
        # RTT (Round Trip Time)
        start_time = datetime.now()
        socket.create_connection((host, port), timeout=2)
        rtt = (datetime.now() - start_time).total_seconds() * 1000
        result["network_info"]["rtt_ms"] = rtt

        # MTU Discovery
        if platform.system().lower() != "windows":
            cmd = f"ping -M do -s 1472 -c 1 {host}"
            try:
                subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
                result["network_info"]["mtu"] = "1500 or larger"
            except Exception:
                result["network_info"]["mtu"] = "Less than 1500"

    except Exception as e:
        result["network_info"]["error"] = str(e)

    # Server Fingerprinting
    try:
        response = requests.get(f"http://{host}:{port}", timeout=timeout)
        result["server_fingerprint"] = {
            "server": response.headers.get("Server", "Not disclosed"),
            "powered_by": response.headers.get("X-Powered-By", "Not disclosed"),
            "content_type": response.headers.get("Content-Type", "Not available"),
            "technologies": response.headers.get("X-Generated-By", "Not available"),
        }
    except Exception as e:
        result["server_fingerprint"]["error"] = str(e)

    return result

def pretty_print_status(status: dict):
    """Pretty print the server status dictionary"""
    print(json.dumps(status, indent=4, sort_keys=True, cls=DateTimeEncoder))

def save_to_file(status: dict, filename: str):
    """Save the results to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(status, f, indent=4, sort_keys=True, cls=DateTimeEncoder)

# Example Usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script.py <host> <port>")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    # Disable SSL warnings
    urllib3.disable_warnings()

    print(f"\nChecking server status for {host}:{port}...")
    status = check_server_status(host, port)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"server_check_{host}_{timestamp}.json"
    save_to_file(status, filename)

    print(f"\nResults saved to {filename}")

    # Print summary
    print("\nSummary:")
    print(f"Server: {host}:{port}")
    print(f"Reachable: {status['basic_info']['reachable']}")
    print(f"IP Address: {status['basic_info']['ip_address']}")
    if status['ssl_details'] and 'error' not in status['ssl_details']:
        print(f"SSL Valid Until: {status['ssl_details']['not_valid_after']}")
    if status['geolocation'] and isinstance(status['geolocation'], dict):
        print(f"Location: {status['geolocation'].get('city', 'Unknown')}, "
              f"{status['geolocation'].get('country', 'Unknown')}")
