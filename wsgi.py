from app import app  # Imports your Flask app instance

if __name__ == "__main__":
    app.run()  # Only runs with 'python wsgi.py', not used by WSGI servers like gunicorn
