import sys
import time
import requests
import argparse

def test_render_service(base_url: str):
    base_url = base_url.rstrip("/")
    print("\n" + "=" * 60)
    print("      ENTERPRISE-IQ RENDER LIVE DEPLOYMENT VERIFICATION     ")
    print("=" * 60)
    print(f"Target URL: {base_url}")

    # 1. Test Streamlit native health check
    health_url = f"{base_url}/_stcore/health"
    print(f"\n[Test 1] Probing Streamlit health endpoint ({health_url})...")
    try:
        t0 = time.time()
        res = requests.get(health_url, timeout=30)
        latency = round((time.time() - t0) * 1000, 2)
        if res.status_code == 200:
            print(f"  ✅ Health check PASSED: Status 200 OK (Latency: {latency} ms)")
        else:
            print(f"  ❌ Health check FAILED: Status {res.status_code} ({res.text[:100]})")
            sys.exit(1)
    except Exception as e:
        print(f"  ❌ Health check FAILED: Connection error ({e})")
        print("  Tip: Render free tier services spin down after inactivity. The first request may take ~30-50s to wake up.")
        sys.exit(1)

    # 2. Test Main Page UI accessibility
    print(f"\n[Test 2] Probing Main Application Page ({base_url})...")
    try:
        t0 = time.time()
        res = requests.get(base_url, timeout=30)
        latency = round((time.time() - t0) * 1000, 2)
        if res.status_code == 200 and "EnterpriseIQ" in res.text:
            print(f"  ✅ Main Page PASSED: EnterpriseIQ UI loaded successfully (Latency: {latency} ms)")
        elif res.status_code == 200:
            print(f"  ✅ Main Page HTTP 200 OK (Latency: {latency} ms)")
        else:
            print(f"  ❌ Main Page FAILED: Status {res.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"  ❌ Main Page probe FAILED: ({e})")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("      ALL RENDER LIVE CHECKS COMPLETED SUCCESSFULLY!        ")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify EnterpriseIQ on Render")
    parser.add_argument("url", nargs="?", default="http://localhost:8501", help="Public URL of the Render service")
    args = parser.parse_args()
    test_render_service(args.url)
