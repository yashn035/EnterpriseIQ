import time
import json
import requests
from pyngrok import ngrok, conf

def start_public_tunnels():
    # Connect ngrok HTTP tunnel to port 8000 (FastAPI API)
    api_tunnel = ngrok.connect(8000, "http")
    api_public_url = api_tunnel.public_url.replace("http://", "https://")

    # Connect ngrok HTTP tunnel to port 8501 (Streamlit Web UI)
    ui_tunnel = ngrok.connect(8501, "http")
    ui_public_url = ui_tunnel.public_url.replace("http://", "https://")

    print("\n===================================================================")
    print("🚀 ENTERPRISEIQ PUBLIC INTERNET URLS")
    print("===================================================================")
    print(f"⚡ FastAPI Public REST API URL:  {api_public_url}")
    print(f"📚 Swagger Interactive API Docs: {api_public_url}/docs")
    print(f"🤖 Streamlit Public Web UI URL: {ui_public_url}")
    print("===================================================================\n")

    # Write URLs to scratch file for record
    tunnel_data = {
        "api_url": api_public_url,
        "ui_url": ui_public_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open("data/public_urls.json", "w", encoding="utf-8") as f:
        json.dump(tunnel_data, f, indent=2)

    # Keep script running to preserve ngrok tunnels
    while True:
        time.sleep(30)

if __name__ == "__main__":
    start_public_tunnels()
