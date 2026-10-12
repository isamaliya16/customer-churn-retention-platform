"""WSGI entrypoint.

Run:  python -m flask --app wsgi run
"""

import os

from app import create_app

app = create_app(os.environ.get("FLASK_ENV") or "development")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
