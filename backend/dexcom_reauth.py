#!/usr/bin/env python3
"""
Re-authenticate with Dexcom OAuth to get fresh tokens.

This will:
1. Start a local HTTPS server
2. Open your browser to Dexcom login
3. Capture the auth code and exchange for tokens
4. Save tokens to tokens.json

Usage:
    python dexcom_reauth.py
"""
import http.server
import ssl
import threading
import requests
import json
import webbrowser
import time
import os
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# === CONFIGURATION (from your existing app) ===
CLIENT_ID = "wGj7Z6aHqQkoBdtnsDXXfkT1ZG03G8jZ"
CLIENT_SECRET = "602yBKHyd6QBO1Yj"
REDIRECT_URI = "https://localhost:7000"
BASE_URL = "https://api.dexcom.com"

# Save tokens to both locations
TOKEN_FILES = [
    Path("/home/jadericdawson/Documents/AI/dexcom_reader_ML_complete/tokens.json"),
    Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/dexcom_tokens.json"),
]

# SSL certs from original app
CERT_DIR = Path("/home/jadericdawson/Documents/AI/dexcom_reader_ML_complete/certs")

tokens = {}
auth_complete = threading.Event()


class OAuthHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

    def do_GET(self):
        global tokens
        parsed_url = urlparse(self.path)
        query = parse_qs(parsed_url.query)

        if "code" in query:
            auth_code = query["code"][0]
            print(f"\n✓ Got authorization code")

            # Exchange for tokens
            if exchange_auth_code(auth_code):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>&#x2705; Success!</h1>
                    <p>Dexcom authentication complete. You can close this window.</p>
                    </body></html>
                """)
                auth_complete.set()
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Token exchange failed")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing auth code")


def exchange_auth_code(code: str) -> bool:
    global tokens
    token_url = f"{BASE_URL}/v2/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }

    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        save_tokens()
        return True
    else:
        print(f"✗ Token exchange failed: {response.text}")
        return False


def save_tokens():
    for token_file in TOKEN_FILES:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(token_file, "w") as f:
            json.dump(tokens, f, indent=2)
        print(f"✓ Saved tokens to {token_file}")


def start_https_server():
    httpd = http.server.HTTPServer(("localhost", 7000), OAuthHandler)

    # Use existing certs
    cert_file = CERT_DIR / "cert.pem"
    key_file = CERT_DIR / "key.pem"

    if not cert_file.exists() or not key_file.exists():
        print(f"✗ SSL certs not found in {CERT_DIR}")
        print("  Creating self-signed certs...")
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        os.system(f'openssl req -x509 -newkey rsa:2048 -keyout {key_file} -out {cert_file} -days 365 -nodes -subj "/CN=localhost"')

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_file), str(key_file))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print("🌐 Listening on https://localhost:7000 ...")
    httpd.serve_forever()


def test_connection():
    """Test the tokens by fetching latest glucose."""
    from datetime import datetime, timedelta

    if not tokens.get("access_token"):
        return False

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=30)

    params = {
        "startDate": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDate": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    response = requests.get(f"{BASE_URL}/v2/users/self/egvs", headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        readings = data.get("egvs", [])
        if readings:
            latest = readings[0]
            age_min = (datetime.utcnow() - datetime.fromisoformat(latest["systemTime"].replace("Z", "+00:00").replace("+00:00", ""))).total_seconds() / 60
            print(f"\n🩸 Latest BG: {latest['value']} mg/dL ({latest.get('trend', '?')})")
            print(f"   Time: {latest['displayTime']}")
            print(f"   Age: {age_min:.1f} minutes")
            return True
    return False


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Dexcom OAuth Re-Authentication")
    print("=" * 50)

    # Start server in background
    server_thread = threading.Thread(target=start_https_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    # Build auth URL
    auth_url = (
        f"{BASE_URL}/v2/oauth2/login?"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=offline_access"
    )

    print(f"\n📱 Opening browser for Dexcom login...")
    print(f"   If browser doesn't open, visit:\n   {auth_url}\n")

    webbrowser.open(auth_url)

    # Wait for auth
    print("⏳ Waiting for authentication...")
    auth_complete.wait(timeout=120)

    if tokens:
        print("\n" + "=" * 50)
        print("✓ Authentication successful!")
        print("=" * 50)

        # Test it
        print("\nTesting connection...")
        if test_connection():
            print("\n✓ All good! Tokens are working.")
        else:
            print("\n⚠ Tokens saved but test failed.")
    else:
        print("\n✗ Authentication timed out or failed.")
