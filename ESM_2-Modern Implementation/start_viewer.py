#!/usr/bin/env python3
"""
Simple HTTP server to view protein structures.
Run this script and open http://localhost:8000/visualize_structure.html in your browser.
"""

import http.server
import socketserver

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

print(f"🧬 Starting Protein Structure Viewer Server...")
print(f"📡 Server running at: http://localhost:{PORT}")
print(f"🌐 Open this URL in your browser: http://localhost:{PORT}/visualize_structure.html")
print(f"⏹️  Press Ctrl+C to stop the server\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped.")
