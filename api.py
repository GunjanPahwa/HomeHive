# api.py (updated)
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import sqlite3
from pathlib import Path
import os
import time
import json

app = Flask(__name__)
CORS(app)

# Define absolute paths to your database files
BASE_DIR = Path(__file__).parent
TICKETS_DB = os.path.join(BASE_DIR, "instance", "tickets.db")
EVENTS_DB = os.path.join(BASE_DIR, "instance", "events.db")

# Ensure the instance directory exists
os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)

def init_db():
    """Initialize database tables if they don't exist"""
    # Initialize tickets database
    conn = sqlite3.connect(TICKETS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            user_id INTEGER,
            username TEXT,
            description TEXT,
            location TEXT,
            priority TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_voice_request BOOLEAN DEFAULT FALSE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            last_seen TIMESTAMP,
            tts_enabled INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()
    
    # Initialize events database
    conn = sqlite3.connect(EVENTS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            creator_id INTEGER NOT NULL,
            creator_name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at DATETIME NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS polls (
            poll_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            options TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            creator_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            votes TEXT,
            total_votes INTEGER DEFAULT 0,
            telegram_poll_id TEXT
        )
    """)
    conn.commit()
    conn.close()

def fetch_data(db_path, query, params=None):
    """Improved database fetch with error handling"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return dictionaries instead of tuples
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return data
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

@app.route("/api/tickets")
def get_tickets():
    """Get tickets with pagination and filtering"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        status = request.args.get('status')
        priority = request.args.get('priority')
        
        offset = (page - 1) * per_page
        
        query = "SELECT * FROM tickets WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        if priority:
            query += " AND priority = ?"
            params.append(priority)
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        tickets = fetch_data(TICKETS_DB, query, params)
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM tickets"
        total = fetch_data(TICKETS_DB, count_query)[0]['COUNT(*)']
        
        return jsonify({
            "data": tickets,
            "total": total,
            "page": page,
            "per_page": per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/events")
def get_events():
    """Get events with pagination and filtering"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        status = request.args.get('status')
        
        offset = (page - 1) * per_page
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        events = fetch_data(EVENTS_DB, query, params)
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM events"
        total = fetch_data(EVENTS_DB, count_query)[0]['COUNT(*)']
        
        return jsonify({
            "data": events,
            "total": total,
            "page": page,
            "per_page": per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/dashboard-data")
def dashboard_data():
    """Get dashboard statistics"""
    try:
        # Get ticket counts by priority
        priority_counts = fetch_data(TICKETS_DB, """
            SELECT 
                SUM(CASE WHEN priority = 'P0' THEN 1 ELSE 0 END) as critical,
                SUM(CASE WHEN priority = 'P1' THEN 1 ELSE 0 END) as high,
                SUM(CASE WHEN priority = 'P2' THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN priority = 'P3' THEN 1 ELSE 0 END) as low
            FROM tickets
            WHERE status = 'Open'
        """)[0]
        
        # Get high priority tickets
        tickets = fetch_data(TICKETS_DB, """
            SELECT 
                ticket_id as id,
                description,
                location,
                priority,
                status,
                created_at as created
            FROM tickets
            WHERE priority IN ('P0', 'P1') AND status = 'Open'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        # Get active polls
        polls = fetch_data(EVENTS_DB, """
            SELECT 
                poll_id as id,
                question,
                created_at as created
            FROM polls
            WHERE status = 'active'
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        return jsonify({
            "priority_counts": priority_counts,
            "tickets": tickets,
            "polls": polls
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stream')
def stream():
    """Server-Sent Events for real-time updates"""
    def event_stream():
        while True:
            try:
                # Get current counts
                tickets_count = fetch_data(TICKETS_DB, "SELECT COUNT(*) FROM tickets")[0]['COUNT(*)']
                events_count = fetch_data(EVENTS_DB, "SELECT COUNT(*) FROM events")[0]['COUNT(*)']
                polls_count = fetch_data(EVENTS_DB, "SELECT COUNT(*) FROM polls")[0]['COUNT(*)']
                
                data = {
                    "tickets": tickets_count,
                    "events": events_count,
                    "polls": polls_count
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            time.sleep(5)
    
    return Response(event_stream(), mimetype="text/event-stream")
@app.route('/')
def home():
    return """
    <h1>HomeHive API is running!</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/api/tickets">/api/tickets</a></li>
        <li><a href="/api/events">/api/events</a></li>
        <li><a href="/api/dashboard-data">/api/dashboard-data</a></li>
    </ul>
    """
@app.route('/api/events', methods=['POST'])
def create_event():
    try:
        event_data = request.get_json()
        
        # Basic validation
        required_fields = ['name', 'dateTime', 'location']
        for field in required_fields:
            if field not in event_data or not event_data[field]:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate event ID
        event_id = f"EVT{int(time.time())}"
        
        # Insert into database
        conn = sqlite3.connect(EVENTS_DB)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (event_id, name, description, creator_id, creator_name, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id,
            event_data['name'],
            event_data.get('description', ''),
            0,  # Admin user ID
            'Admin',  # Creator name
            'planned',
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            "id": event_id,
            "name": event_data['name'],
            "dateTime": event_data['dateTime'],
            "location": event_data['location'],
            "description": event_data.get('description', ''),
            "status": "planned",
            "attendees": 0,
            "maxAttendees": 50
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()  # Initialize database on startup
    app.run(debug=True, port=5000)