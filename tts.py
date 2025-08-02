import os
import uuid
import json
import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
from functools import lru_cache
import time
import sqlite3
from pathlib import Path
import tempfile
import io
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.constants import ChatAction
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, Poll
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, 
    ContextTypes, filters, PollAnswerHandler, PollHandler
)

# AI/NLP
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ Gemini AI not available - AI features disabled")
    GEMINI_AVAILABLE = False

# Voice Processing
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    import requests
    VOICE_AVAILABLE = True
except ImportError:
    print("⚠️ Voice processing not available - installing: pip install SpeechRecognition pydub requests")
    VOICE_AVAILABLE = False
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️ gTTS not available - install with: pip install gTTS")
    TTS_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)
class TextToSpeechProcessor:
    """Text-to-Speech processor using gTTS"""
    
    def __init__(self):
        self.available = TTS_AVAILABLE
        if self.available:
            logger.info("✅ TTS features enabled (gTTS)")
        else:
            logger.warning("⚠️ TTS features disabled - missing gTTS")
    
    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using gTTS with better error handling"""
        if not self.available:
            logger.warning("TTS not available - gTTS not installed")
            return None
            
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        # Clean text for better speech synthesis
        cleaned_text = self._clean_text_for_speech(text)
        
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:497] + "..."
        
        try:
            # Generate speech with timeout
            import asyncio
            import concurrent.futures
            
            def generate_tts():
                tts = gTTS(text=cleaned_text, lang='en', slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            
            # Run with timeout to prevent hanging
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, generate_tts)
                audio_data = await asyncio.wait_for(future, timeout=10.0)
                
            logger.info(f"TTS generated successfully for text: {cleaned_text[:50]}...")
            return audio_data
            
        except asyncio.TimeoutError:
            logger.error("TTS generation timed out after 10 seconds")
            return None
        except ImportError:
            logger.error("gTTS not installed - install with: pip install gTTS")
            return None
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            return None
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'_(.*?)_', r'\1', text)        # Underline
        
        # Remove emojis that might not be spoken well
        text = re.sub(r'[🎤🔧📊🗳️📅💡⚠️❌✅🚨📝📍⏰🎉👥👤📋🔄🆕❓]', '', text)
        
        # Replace common symbols with words
        replacements = {
            '&': 'and',
            '@': 'at',
            '#': 'number',
            '%': 'percent',
            '/': ' slash ',
            '|': ' or ',
            '•': '. ',
            '→': ' to ',
            '←': ' from '
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
        
        # Clean up multiple spaces and line breaks
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    async def get_user_tts_settings(self, user_id: int) -> Dict:
        """Get user's TTS preferences with proper user creation"""
        try:
            # First ensure user exists in database
            await self.log_user_simple(user_id)
            
            cursor = self.ticket_db.execute(
                "SELECT tts_enabled FROM users WHERE user_id = ?", 
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return {'tts_enabled': bool(result[0])}
            else:
                # Create user with default settings
                self.ticket_db.execute(
                    "INSERT OR IGNORE INTO users (user_id, tts_enabled) VALUES (?, ?)",
                    (user_id, 1),
                    commit=True
                )
                return {'tts_enabled': True}
                
        except Exception as e:
            logger.error(f"Error getting TTS settings for user {user_id}: {e}")
            return {'tts_enabled': True}

    async def log_user_simple(self, user_id: int):
        """Simple user logging for TTS settings"""
        try:
            query = """
            INSERT OR IGNORE INTO users (user_id, tts_enabled, last_seen)
            VALUES (?, ?, ?)
            """
            params = (user_id, 1, datetime.now().isoformat())
            self.ticket_db.execute(query, params, commit=True)
        except Exception as e:
            logger.error(f"Error logging user {user_id}: {e}")

    # 5. IMPROVED send_response_with_tts method
    async def send_response_with_tts(self, update: Update, text: str, parse_mode: str = None, reply_markup=None):
        """Send text response and optionally include TTS audio with better error handling"""
        try:
            user_id = update.effective_user.id
            
            # Send text message first (always works)
            message = await update.message.reply_text(
                text, 
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            
            # Try TTS if available
            if self.tts_processor.available:
                try:
                    user_settings = await self.get_user_tts_settings(user_id)
                    
                    if user_settings.get('tts_enabled', False):
                        # Show typing indicator
                        await self.application.bot.send_chat_action(
                            chat_id=update.effective_chat.id,
                            action=ChatAction.RECORD_VOICE
                        )
                        
                        # Generate speech with timeout
                        audio_data = await self.tts_processor.text_to_speech(text)
                        
                        if audio_data and len(audio_data) > 0:
                            # Send as voice message
                            await update.message.reply_voice(
                                voice=io.BytesIO(audio_data),
                                caption="🔊 Audio version"
                            )
                            logger.info(f"TTS sent successfully to user {user_id}")
                        else:
                            logger.warning(f"TTS generation returned empty audio for user {user_id}")
                            
                except Exception as tts_error:
                    logger.error(f"TTS failed for user {user_id}: {str(tts_error)}")
                    # Don't fail the whole response if TTS fails
                    pass
            else:
                logger.debug("TTS processor not available")
                        
            return message
            
        except Exception as e:
            logger.error(f"Error in send_response_with_tts: {e}")
            # Fallback to regular text message
            try:
                return await update.message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
            except Exception as fallback_error:
                logger.error(f"Fallback message also failed: {fallback_error}")
                raise

    async def debug_tts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug TTS system status"""
        try:
            from gtts import gTTS
            gtts_status = "✅ Installed"
        except ImportError:
            gtts_status = "❌ Not installed - run: pip install gTTS"
        
        # Check processor
        processor_status = "✅ Ready" if self.tts_processor.available else "❌ Not ready"
        try:
            user_id = update.effective_user.id
            
            # Check gTTS availability
            tts_status = "✅ Available" if TTS_AVAILABLE else "❌ Not installed"
            processor_status = "✅ Ready" if self.tts_processor.available else "❌ Not ready"
            
            # Check user settings
            user_settings = await self.get_user_tts_settings(user_id)
            user_tts_status = "✅ Enabled" if user_settings.get('tts_enabled') else "❌ Disabled"
            
            # Test TTS generation
            test_result = "❓ Testing..."
            try:
                test_audio = await self.tts_processor.text_to_speech("This is a test")
                test_result = f"✅ Success ({len(test_audio)} bytes)" if test_audio else "❌ Failed"
            except Exception as e:
                test_result = f"❌ Error: {str(e)[:50]}"
            
            debug_info = f"""🔍 **TTS Debug Information**

    **System Status:**
    • gTTS Library: {tts_status}
    • TTS Processor: {processor_status}
    • User TTS Setting: {user_tts_status}
    • Test Generation: {test_result}

    **User Info:**
    • User ID: {user_id}
    • Username: {update.effective_user.username or 'None'}

    **Installation Check:**
    ```
    pip install gTTS
    ```

    **Troubleshooting:**
    1. Check internet connection
    2. Verify gTTS installation
    3. Try /tts on to enable
    4. Use /speak test message"""
            
            await update.message.reply_text(debug_info, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Debug command failed: {str(e)}")
    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Simple test command to verify handlers are working"""
        await update.message.reply_text("✅ Test command is working! TTS handlers should work too.")
class TicketDatabase:
    """SQLite database handler for tickets and maintenance"""
    def __init__(self, db_path="tickets.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database with required tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            last_seen TIMESTAMP
        )
        """)
        
        # Tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            description TEXT,
            location TEXT,
            priority TEXT,
            status TEXT,
            created_at TIMESTAMP,
            is_voice_request INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN tts_enabled INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            description TEXT,
            location TEXT,
            priority TEXT,
            status TEXT,
            created_at TIMESTAMP,
            is_voice_request INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
        self.conn.commit()
    
    def execute(self, query, params=(), commit=False):
        """Execute a SQL query"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            if commit:
                self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

class EventDatabase:
    """SQLite database handler for events and polls"""
    def __init__(self, db_path="events.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Initialize database tables"""
        self.cur.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            creator_id INTEGER NOT NULL,
            creator_name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at DATETIME NOT NULL
        )
        ''')

        self.cur.execute('''
        CREATE TABLE IF NOT EXISTS event_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            response TEXT CHECK(response IN ('positive', 'negative')),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        )
        ''')
        
        self.cur.execute('''
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
        ''')
        
        self.conn.commit()

    def create_event(self, event_data: dict) -> dict:
        """Create a new event"""
        event_id = f"EVENT{int(datetime.now().timestamp())}"
        event_data.update({
            'event_id': event_id,
            'created_at': datetime.now().isoformat()
        })

        self.cur.execute('''
        INSERT INTO events (event_id, name, description, creator_id, creator_name, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_data['event_id'],
            event_data.get('name', 'Unnamed Event'),
            event_data.get('description', ''),
            event_data['creator_id'],
            event_data.get('creator_name', 'Anonymous'),
            event_data.get('status', 'pending'),
            event_data['created_at']
        ))
        self.conn.commit()
        return event_data

    def get_events(self, status: str = None) -> list:
        """Get all events (optionally filtered by status)"""
        query = "SELECT * FROM events"
        params = ()
        
        if status:
            query += " WHERE status = ?"
            params = (status,)
            
        self.cur.execute(query, params)
        columns = [col[0] for col in self.cur.description]
        return [dict(zip(columns, row)) for row in self.cur.fetchall()]

    def create_poll(self, poll_data: Dict) -> Dict:
        """Create a new poll"""
        poll_id = f"POLL{int(time.time())}"
        poll_data.update({
            'poll_id': poll_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'votes': json.dumps({option: [] for option in poll_data.get('options', ["Yes! 🎉", "No 😕", "Maybe 🤔"])}),
            'total_votes': 0
        })
        
        self.cur.execute('''
        INSERT INTO polls (poll_id, question, options, created_by, creator_name, created_at, status, votes, total_votes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            poll_data['poll_id'],
            poll_data['question'],
            json.dumps(poll_data.get('options', ["Yes! 🎉", "No 😕", "Maybe 🤔"])),
            poll_data['created_by'],
            poll_data['creator_name'],
            poll_data['created_at'],
            poll_data['status'],
            poll_data['votes'],
            poll_data['total_votes']
        ))
        self.conn.commit()
        return poll_data

    def get_poll(self, poll_id: str) -> Optional[Dict]:
        """Get a poll by ID"""
        self.cur.execute('SELECT * FROM polls WHERE poll_id = ?', (poll_id,))
        result = self.cur.fetchone()
        if result:
            columns = [col[0] for col in self.cur.description]
            poll = dict(zip(columns, result))
            poll['options'] = json.loads(poll['options'])
            poll['votes'] = json.loads(poll['votes']) if poll['votes'] else {}
            return poll
        return None

    def get_active_polls(self) -> List[Dict]:
        """Get all active polls"""
        self.cur.execute('''
            SELECT * FROM polls 
            WHERE status = 'active'
            ORDER BY created_at DESC
        ''')
        
        columns = [col[0] for col in self.cur.description]
        polls = []
        for row in self.cur.fetchall():
            poll = dict(zip(columns, row))
            poll['options'] = json.loads(poll['options'])
            poll['votes'] = json.loads(poll['votes']) if poll['votes'] else {}
            polls.append(poll)
        return polls

    def close(self):
        """Close database connection"""
        self.conn.close()

class VoiceProcessor:
    """Simplified voice processing with multiple fallbacks"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust recognizer settings for better performance
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    async def voice_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """Convert voice to text with multiple fallback methods"""
        logger.info("🎤 Starting voice recognition...")
        
        # Try different methods in order
        methods = [
            ("Google Speech API", self._google_recognition),
            ("Offline Sphinx", self._sphinx_recognition)
        ]
        
        for method_name, method in methods:
            try:
                logger.info(f"Trying {method_name}...")
                result = await asyncio.get_event_loop().run_in_executor(
                    None, method, audio_bytes
                )
                
                if result and result.strip():
                    logger.info(f"✅ {method_name} succeeded: '{result[:50]}...'")
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"❌ {method_name} failed: {str(e)}")
                continue
        
        logger.error("All voice recognition methods failed")
        return None

    def _google_recognition(self, audio_bytes: bytes) -> Optional[str]:
        """Google Speech Recognition"""
        temp_files = []
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            # Convert OGG to WAV
            try:
                audio = AudioSegment.from_ogg(temp_path)
            except:
                # Fallback: try as generic audio file
                audio = AudioSegment.from_file(temp_path)
            
            # Optimize for speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export to WAV in memory
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Perform speech recognition
            with sr.AudioFile(wav_buffer) as source:
                # Adjust for ambient noise
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except:
                    pass  # Skip if adjustment fails
                
                audio_data = self.recognizer.record(source)
            
            # Try Google Speech Recognition
            return self.recognizer.recognize_google(audio_data, language='en-US')
                
        except sr.UnknownValueError:
            raise Exception("Could not understand the audio")
        except sr.RequestError as e:
            raise Exception(f"Google Speech API error: {e}")
        except Exception as e:
            raise Exception(f"Audio processing error: {e}")
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _sphinx_recognition(self, audio_bytes: bytes) -> Optional[str]:
        """Offline Sphinx recognition"""
        temp_files = []
        try:
            # Save and convert audio
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            try:
                audio = AudioSegment.from_ogg(temp_path)
            except:
                audio = AudioSegment.from_file(temp_path)
            
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Sphinx recognition
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)
            
            return self.recognizer.recognize_sphinx(audio_data)
                
        except sr.UnknownValueError:
            raise Exception("Sphinx could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Sphinx error: {e}")
        except Exception as e:
            raise Exception(f"Sphinx processing error: {e}")
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

class HomeHiveBuddy:
    def __init__(self, telegram_token: str):
        # Initialize databases
        self.ticket_db = TicketDatabase()
        self.event_db = EventDatabase()
        
        # Initialize voice processor
        self.voice_enabled = VOICE_AVAILABLE
        if self.voice_enabled:
            self.voice_processor = VoiceProcessor()
            logger.info("✅ Voice features enabled")
        else:
            logger.warning("⚠️ Voice features disabled - missing dependencies")
        self.tts_processor = TextToSpeechProcessor()
        # Initialize counters and keywords
        self.ticket_counter = 1000
        self.poll_counter = 1000
        self.urgent_keywords = ['leak', 'flooding', 'fire', 'emergency', 'urgent', 'broken', 'not working']
        self.maintenance_keywords = ['repair', 'fix', 'broken', 'not working', 'issue', 'problem', 'maintenance']
        self.event_keywords = ['event', 'activity', 'party', 'meeting', 'gathering']
        
        # Initialize AI
        self._initialize_ai()
        
        # Initialize Telegram Bot
        self.bot_token = telegram_token
        self.application = (
            Application.builder()
            .token(telegram_token)
            .read_timeout(30)
            .write_timeout(30)
            .build()
        )
        
        # Setup handlers
        self.setup_handlers()

    def _initialize_ai(self):
        """Initialize AI with proper error handling"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                try:
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                except:
                    try:
                        self.gemini_model = genai.GenerativeModel('gemini-pro')
                    except:
                        logger.warning("Could not initialize Gemini model")
                        self.gemini_model = None
                
                if self.gemini_model:
                    logger.info("✅ Gemini AI initialized")
            except Exception as e:
                logger.warning(f"⚠️ Gemini AI initialization failed: {e}")
                self.gemini_model = None

    def setup_handlers(self):
        """Setup all command and message handlers"""
        # Basic commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("events", self.events_command))
        self.application.add_handler(CommandHandler("maintenance", self.maintenance_command))
        
        # TTS COMMANDS - ADD THESE MISSING HANDLERS
        self.application.add_handler(CommandHandler("tts", self.tts_command))
        self.application.add_handler(CommandHandler("speak", self.speak_command))
        
        # Poll commands
        self.application.add_handler(CommandHandler("createpoll", self.create_event_poll_command))
        self.application.add_handler(CommandHandler("activepolls", self.show_active_polls))
        self.application.add_handler(CommandHandler("pollresults", self.show_poll_results))
        
        # Voice handlers (if available)
        if self.voice_enabled:
            self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
            self.application.add_handler(MessageHandler(filters.AUDIO, self.handle_voice))
        
        # Poll handlers
        self.application.add_handler(PollAnswerHandler(self.handle_poll_answer))
        
        # Callback handler
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # General text handler (must come last)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def get_user_tts_settings(self, user_id: int) -> Dict:
        """Get user's TTS preferences"""
        try:
            cursor = self.ticket_db.execute(
                "SELECT tts_enabled FROM users WHERE user_id = ?", 
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return {'tts_enabled': bool(result[0])}
            else:
                # Default settings for new users
                return {'tts_enabled': True}
        except Exception as e:
            logger.error(f"Error getting TTS settings: {e}")
            return {'tts_enabled': True}

    async def update_user_tts_settings(self, user_id: int, tts_enabled: bool):
        """Update user's TTS preferences"""
        try:
            self.ticket_db.execute(
                "UPDATE users SET tts_enabled = ? WHERE user_id = ?",
                (int(tts_enabled), user_id),
                commit=True
            )
        except Exception as e:
            logger.error(f"Error updating TTS settings: {e}")

    async def send_response_with_tts(self, update: Update, text: str, parse_mode: str = None, reply_markup=None):
        """Send text response and optionally include TTS audio"""
        try:
            user_id = update.effective_user.id
            user_settings = await self.get_user_tts_settings(user_id)
            
            # Send text message first
            message = await update.message.reply_text(
                text, 
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            
            # Send TTS audio if enabled and available
            if user_settings['tts_enabled'] and self.tts_processor.available:
                try:
                    # Show typing indicator
                    await self.application.bot.send_chat_action(
                        chat_id=update.effective_chat.id,
                        action=ChatAction.RECORD_VOICE
                    )
                    
                    # Generate speech
                    audio_data = await self.tts_processor.text_to_speech(text)
                    
                    if audio_data:
                        # Send as voice message
                        await update.message.reply_voice(
                            voice=io.BytesIO(audio_data),
                            caption="🔊 Audio version"
                        )
                        
                except Exception as e:
                    logger.error(f"TTS failed for user {user_id}: {e}")
                    # Don't fail the whole response if TTS fails
                    pass
                    
            return message
            
        except Exception as e:
            logger.error(f"Error in send_response_with_tts: {e}")
            # Fallback to regular text message
            return await update.message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)

    # TTS COMMAND HANDLERS
    async def tts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle TTS on/off for user"""
        try:
            user_id = update.effective_user.id
            current_settings = await self.get_user_tts_settings(user_id)
            
            if context.args:
                arg = context.args[0].lower()
                if arg in ['on', 'enable', 'true', '1']:
                    new_setting = True
                elif arg in ['off', 'disable', 'false', '0']:
                    new_setting = False
                else:
                    await update.message.reply_text(
                        "❓ Usage: /tts [on/off]\n\n"
                        "Examples:\n"
                        "• /tts on - Enable voice responses\n"
                        "• /tts off - Disable voice responses\n"
                        "• /tts - Show current status"
                    )
                    return
            else:
                # Toggle current setting
                new_setting = not current_settings['tts_enabled']
            
            # Update setting
            await self.update_user_tts_settings(user_id, tts_enabled=new_setting)
            
            status = "enabled" if new_setting else "disabled"
            emoji = "🔊" if new_setting else "🔇"
            
            response = f"{emoji} **Text-to-Speech {status.capitalize()}**\n\n"
            
            if new_setting:
                if self.tts_processor.available:
                    response += "✅ Voice responses will now be sent with text messages.\n"
                    response += "💡 Use /speak to test the TTS system."
                else:
                    response += "⚠️ No TTS engine available. Install gTTS with:\n"
                    response += "`pip install gTTS`"
            else:
                response += "You will only receive text responses now.\n"
                response += "Use /tts on to re-enable voice responses."
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"TTS command error: {e}")
            await update.message.reply_text("❌ Error updating TTS settings.")

    async def speak_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Convert provided text to speech"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "🔊 **Text-to-Speech Converter**\n\n"
                    "**Usage:** /speak <text to convert>\n\n"
                    "**Examples:**\n"
                    "• /speak Hello, how are you today?\n"
                    "• /speak Your maintenance ticket has been created\n"
                    "• /speak The event is scheduled for tomorrow\n\n"
                    "💡 Use /tts to enable automatic voice responses",
                    parse_mode='Markdown'
                )
                return
            
            text_to_speak = ' '.join(context.args)
            
            if len(text_to_speak) > 500:
                await update.message.reply_text(
                    "❌ Text too long! Please limit to 500 characters.\n"
                    f"Your text: {len(text_to_speak)} characters"
                )
                return
            
            # Show processing message
            processing_msg = await update.message.reply_text(
                "🔊 **Converting text to speech...**\n\n"
                f"📝 Text: \"{text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}\"\n"
                "⏳ Please wait...",
                parse_mode='Markdown'
            )
            
            # Show voice recording indicator
            await self.application.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.RECORD_VOICE
            )
            
            # Generate speech
            audio_data = await self.tts_processor.text_to_speech(text_to_speak)
            
            if audio_data:
                # Delete processing message
                await processing_msg.delete()
                
                # Send voice message
                await update.message.reply_voice(
                    voice=io.BytesIO(audio_data),
                    caption=f"🔊 **Audio conversion complete**\n📝 \"{text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}\"",
                    parse_mode='Markdown'
                )
                
                # Send success message
                await update.message.reply_text(
                    f"✅ **Voice generated successfully!**\n"
                    f"💡 Change TTS settings with /tts",
                    parse_mode='Markdown'
                )
            else:
                await processing_msg.edit_text(
                    "❌ **Speech generation failed**\n\n"
                    "🔧 **Troubleshooting:**\n"
                    "• Check internet connection\n"
                    "• Install gTTS: `pip install gTTS`\n"
                    "• Contact support if problem persists",
                    parse_mode='Markdown'
                )
            
        except Exception as e:
            logger.error(f"Speak command error: {e}")
            await update.message.reply_text(
                f"❌ **Error generating speech**\n\n"
                f"🐛 Error details: {str(e)[:100]}...\n\n"
                f"💡 Try /tts to check TTS status",
                parse_mode='Markdown'
            )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with voice instructions"""
        voice_status = "✅ Voice Assistant Ready!" if self.voice_enabled else "⚠️ Voice features disabled"
        
        welcome_msg = f"""🎤 *HomeHive Buddy - Voice Assistant* 🏠

{voice_status}

*How to use:*
📱 *Text:* Type your requests normally
🎤 *Voice:* Send voice messages for hands-free operation

*Try saying:*
• "My air conditioner is not working"
• "There's a leak in apartment 2B"
• "Schedule maintenance for my unit"
• "Create a poll about movie night"

*Commands:*
/help - Show all commands
/status - Check your tickets
/events - View upcoming events

Ready to help! 🤖"""
        
        # Create keyboard with voice prompt
        if self.voice_enabled:
            keyboard = [[KeyboardButton("🎤 Try Voice Message")]]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        else:
            reply_markup = None
        
        await update.message.reply_text(
            welcome_msg,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        await self.log_user(update.effective_user)

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages with improved error handling"""
        if not self.voice_enabled:
            await update.message.reply_text(
                "❌ Voice features are not available. Please install required packages:\n"
                "`pip install SpeechRecognition pydub requests`"
            )
            return
        
        try:
            # Show processing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )
            
            # Get voice file
            voice_file = await (update.message.voice or update.message.audio).get_file()
            audio_bytes = await voice_file.download_as_bytearray()
            
            # Show processing message
            try:
                processing_msg = await update.message.reply_text(
                    "🎤 Processing your voice message...\n\n"
                    "⏳ Converting speech to text...",
                    reply_markup=ReplyKeyboardRemove()
                )
            except Exception as e:
                logger.error(f"Error sending processing message: {e}")
                processing_msg = None
            
            # Convert voice to text
            text = await self.voice_processor.voice_to_text(bytes(audio_bytes))
            
            if not text:
                error_msg = (
                    "❌ Sorry, I couldn't understand your voice message.\n\n"
                    "**Tips for better recognition:**\n"
                    "• Speak clearly and slowly\n"
                    "• Reduce background noise\n"
                    "• Hold phone closer to your mouth\n"
                    "• Try again or type your message\n\n"
                    "🔄 Feel free to try again!"
                )
                if processing_msg:
                    try:
                        await processing_msg.edit_text(error_msg, parse_mode='Markdown')
                    except Exception as edit_error:
                        logger.warning(f"Couldn't edit message: {edit_error}")
                        await update.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Show what we understood
            result_msg = f"🎤 **I heard:** \"{text}\"\n\n🔄 Processing your request..."
            if processing_msg:
                try:
                    await processing_msg.edit_text(result_msg, parse_mode='Markdown')
                except Exception as edit_error:
                    logger.warning(f"Couldn't edit message: {edit_error}")
                    await update.message.reply_text(result_msg, parse_mode='Markdown')
                    processing_msg = None  # Don't use this message further
            
            # Process the voice command
            await self.process_voice_command(update, text, processing_msg)
                
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            error_msg = (
                "❌ Error processing voice message. Please try again or use text.\n\n"
                f"Error: {str(e)[:100]}..."
            )
            try:
                if processing_msg:
                    await processing_msg.edit_text(error_msg)
                else:
                    await update.message.reply_text(error_msg)
            except Exception as final_error:
                logger.error(f"Couldn't send error message: {final_error}")

    async def process_voice_command(self, update: Update, text: str, processing_msg):
        """Process transcribed voice command with better message handling"""
        try:
            text_lower = text.lower()
            
            # Delete processing message if it exists
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception as delete_error:
                    logger.warning(f"Couldn't delete processing message: {delete_error}")
            
            # Show confirmation of what was heard
            try:
                heard_msg = await update.message.reply_text(
                    f"🎤 **Heard:** \"{text}\"\n\n🔄 Processing...",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Couldn't send heard message: {e}")
                heard_msg = None
            
            # Route to appropriate handler
            if any(keyword in text_lower for keyword in self.maintenance_keywords):
                if heard_msg:
                    try:
                        await heard_msg.delete()
                    except:
                        pass
                await self.handle_maintenance_request(update, text, is_voice=True)
            elif any(keyword in text_lower for keyword in self.event_keywords):
                if heard_msg:
                    try:
                        await heard_msg.delete()
                    except:
                        pass
                await self.handle_event_request(update, text, is_voice=True)
            elif self.is_poll_creation_request(text_lower):
                if heard_msg:
                    try:
                        await heard_msg.delete()
                    except:
                        pass
                question = self.extract_poll_question(text)
                await self.create_poll_from_text(update, question)
            else:
                if heard_msg:
                    try:
                        await heard_msg.delete()
                    except:
                        pass
                await self.handle_general_query(update, text, is_voice=True)
                    
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            await self.send_response_with_tts(
                update,
                f"❌ Error processing voice command: {str(e)[:100]}..."
            )
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages with improved logic"""
        text = update.message.text.lower()
        
        # Handle TTS test request
        if text in ["🔊 test text-to-speech", "test tts", "try tts"]:
            await self.test_tts_for_user(update)
            return
        
        # Check for poll creation requests
        if self.is_poll_creation_request(text):
            question = self.extract_poll_question(update.message.text)
            await self.create_poll_from_text(update, question)
            return
        
        # Check for maintenance requests
        if any(keyword in text for keyword in self.maintenance_keywords):
            await self.handle_maintenance_request(update, update.message.text)
            return
        
        # Check for event requests
        if any(keyword in text for keyword in self.event_keywords):
            await self.handle_event_request(update, update.message.text)
            return
        
        # Default to general query
        await self.handle_general_query(update, update.message.text)

    async def test_tts_for_user(self, update: Update):
        """Test TTS functionality for user"""
        try:
            user_settings = await self.get_user_tts_settings(update.effective_user.id)
                
            if not self.tts_processor.available:
                await update.message.reply_text(
                    "❌ **No TTS engine available**\n\n"
                    "Install gTTS with:\n"
                    "`pip install gTTS`\n\n"
                    "🔄 Restart the bot after installation",
                    parse_mode='Markdown'
                )
                return
                
            test_text = f"Hello! This is a test of the text-to-speech system. Your TTS is currently {'enabled' if user_settings['tts_enabled'] else 'disabled'}. The system is working perfectly!"
                
            # Show processing
            processing_msg = await update.message.reply_text(
                f"🔊 **Testing TTS System**\n\n"
                f"📝 Text: \"{test_text[:50]}...\"\n"
                f"⏳ Generating audio...",
                parse_mode='Markdown'
            )
                
            await self.application.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.RECORD_VOICE
            )
                
            # Generate speech
            audio_data = await self.tts_processor.text_to_speech(test_text)
                
            if audio_data:
                await processing_msg.delete()
                    
                await update.message.reply_voice(
                    voice=io.BytesIO(audio_data),
                    caption=f"✅ **TTS Test Successful!**",
                    parse_mode='Markdown'
                )
                    
                await update.message.reply_text(
                    f"🎉 **Text-to-Speech is working!**\n\n"
                    f"🔊 Status: {'Enabled' if user_settings['tts_enabled'] else 'Disabled'}\n"
                    f"💡 Use /tts to configure preferences",
                    parse_mode='Markdown'
                )
            else:
                await processing_msg.edit_text(
                    f"❌ **TTS Test Failed**\n\n"
                    f"🔧 Try installing gTTS: `pip install gTTS`",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"TTS test error: {e}")
            await update.message.reply_text("❌ Error testing TTS system.")

    def is_poll_creation_request(self, text: str) -> bool:
        """Check if text is requesting a poll creation"""
        poll_keywords = [
            'create poll', 'make poll', 'start vote', 'should we', 
            'what do you think about', 'poll about', 'vote on'
        ]
        return any(keyword in text for keyword in poll_keywords)

    def extract_poll_question(self, text: str) -> str:
        """Extract poll question from natural language"""
        # Remove common poll request phrases
        text = re.sub(r'(create|make|start)\s+(a\s+)?(poll|vote)\s+(about\s+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'what do you think about\s+', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        # Ensure it ends with punctuation
        if not text.endswith(('?', '!', '.')):
            text += '?'
        
        return text.capitalize()

    async def create_poll_from_text(self, update: Update, question: str):
        """Create a poll from extracted question"""
        try:
            # Generate poll ID
            poll_id = f"POLL{self.poll_counter}"
            self.poll_counter += 1

            # Create default options
            options = ["Yes! 🎉", "No 😕", "Maybe 🤔"]

            # Create poll data
            poll_data = {
                'poll_id': poll_id,
                'question': question,
                'options': options,
                'created_by': update.effective_user.id,
                'creator_name': update.effective_user.first_name or "Unknown",
                'created_at': datetime.now(),
                'status': 'active',
                'votes': {option: [] for option in options},
                'total_votes': 0
            }

            # Save to database
            self.event_db.create_poll(poll_data)

            # Send poll
            poll_message = await update.message.reply_poll(
                question=f"🗳️ {question}",
                options=options,
                is_anonymous=False,
                allows_multiple_answers=False
            )

            # Send confirmation
            await update.message.reply_text(
                f"✅ *Poll Created Successfully!*\n\n"
                f"📊 *Poll ID:* {poll_id}\n"
                f"❓ *Question:* {question}\n\n"
                f"Everyone can now vote! 🗳️",
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"Error creating poll from text: {e}")
            await update.message.reply_text("❌ Sorry, I couldn't create the poll. Please try again.")

    async def handle_maintenance_request(self, update: Update, text: str, is_voice: bool = False):
        """Handle maintenance requests"""
        try:
            user = update.effective_user
            
            # Generate ticket ID
            ticket_id = f"T{self.ticket_counter}"
            self.ticket_counter += 1
            
            # Extract location
            location = "Not specified"
            location_patterns = [
                r'(?i)(room|unit|apt|apartment)\s*(\d+[a-z]?)',
                r'(?i)(floor|level)\s*(\d+)',
                r'(?i)(\d+[a-z]?)\s*(floor|level|room|unit|apt|apartment)'
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, text)
                if match:
                    location = f"{match.group(1).capitalize()} {match.group(2)}"
                    break
            
            # Classify priority
            priority = "P1" if any(word in text.lower() for word in self.urgent_keywords) else "P2"
            
            # Voice indicator
            voice_indicator = "🎤 (Voice Request)" if is_voice else "💬 (Text Request)"
            
            # Create response
            response_text = f"""🎫 *Ticket {ticket_id} Created* {voice_indicator}

📋 *Issue:* {text}
📍 *Location:* {location}
🚨 *Priority:* {priority}
⏰ *Status:* Open
📅 *Created:* {datetime.now().strftime('%Y-%m-%d %H:%M')}

{self._get_eta_message(priority)}"""
            
            # Create action buttons
            keyboard = [
                [InlineKeyboardButton("📊 Check Status", callback_data=f"status_{ticket_id}")],
                [InlineKeyboardButton("❌ Cancel Ticket", callback_data=f"cancel_{ticket_id}")]
            ]
            
            await update.message.reply_text(
                response_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
            # Save ticket to database
            await self.save_ticket({
                'ticket_id': ticket_id,
                'user_id': user.id,
                'username': user.username or user.first_name or "Unknown",
                'description': text,
                'location': location,
                'priority': priority,
                'status': 'Open',
                'created_at': datetime.now().isoformat(),
                'is_voice_request': is_voice
            })
                
        except Exception as e:
            logger.error(f"Maintenance request failed: {e}")
            await update.message.reply_text(
                "⚠️ Couldn't create ticket. Please try again or contact support."
            )

    def _get_eta_message(self, priority: str) -> str:
        """Generate ETA message based on priority"""
        if priority == "P1":
            return "🚨 *Urgent Priority!* Maintenance team notified immediately.\n⏰ *ETA:* 1-2 hours"
        else:
            return "⏱️ *Standard Priority* - Processing during business hours.\n⏰ *ETA:* 4-6 hours (next business day)"

    async def handle_event_request(self, update: Update, text: str, is_voice: bool = False):
        """Handle event-related requests"""
        text_lower = text.lower()
        
        if 'upcoming' in text_lower or 'what' in text_lower or 'show' in text_lower:
            await self.show_upcoming_events(update)
        else:
            await self.show_event_menu(update)

    async def show_upcoming_events(self, update: Update):
        """Show upcoming events"""
        try:
            events = self.event_db.get_events(status='active')
            
            if not events:
                await update.message.reply_text(
                    "📅 *No upcoming events right now.*\n\n"
                    "Want to suggest an event? Just say:\n"
                    "• \"Create a poll about movie night\"\n"
                    "• \"Should we have a barbecue party?\"\n"
                    "• Use /createpoll command",
                    parse_mode='Markdown'
                )
                return
            
            response = "📅 *Upcoming Events:*\n\n"
            
            for i, event in enumerate(events[:5], 1):
                response += f"{i}. 🎉 *{event['name']}*\n"
                if event.get('description'):
                    response += f"   📝 {event['description']}\n"
                try:
                    created_at = datetime.fromisoformat(event['created_at']) if isinstance(event['created_at'], str) else event['created_at']
                    response += f"   📅 {created_at.strftime('%m/%d %H:%M')}\n\n"
                except:
                    response += f"   📅 Recently created\n\n"
            
            response += "💡 *Want to suggest a new event?* Just ask!\n"
            response += "🗳️ Use /createpoll or say \"Create a poll about...\""
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing upcoming events: {e}")
            await update.message.reply_text("❌ Error fetching events.")

    async def show_event_menu(self, update: Update):
        """Show event menu with options"""
        menu_text = """🎉 *Event Center*

*Available Actions:*
📅 View upcoming events
🗳️ Create event polls
📊 Check poll results

*Try saying:*
• "What events are coming up?"
• "Create a poll about game night"
• "Show me the polls"

*Commands:*
/events - View upcoming events
/createpoll <question> - Create a poll
/activepolls - View active polls"""

        keyboard = [
            [InlineKeyboardButton("📅 Upcoming Events", callback_data="show_events")],
            [InlineKeyboardButton("🗳️ Active Polls", callback_data="show_polls")],
            [InlineKeyboardButton("📊 Create Poll", callback_data="create_poll_prompt")]
        ]
        
        await update.message.reply_text(
            menu_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_general_query(self, update: Update, text: str, is_voice: bool = False):
        """Handle general queries with context awareness"""
        text_lower = text.lower()
        voice_indicator = "🎤 " if is_voice else ""
        
        # Greeting responses
        greetings = {
            'hello': f"{voice_indicator}Hello! 👋 How can I help you today?",
            'hi': f"{voice_indicator}Hi there! 👋 What can I do for you?",
            'hey': f"{voice_indicator}Hey! 👋 Ready to assist you!",
            'good morning': f"{voice_indicator}Good morning! ☀️ How can I help?",
            'good afternoon': f"{voice_indicator}Good afternoon! 🌅 What do you need?",
            'good evening': f"{voice_indicator}Good evening! 🌆 How can I assist?"
        }
        
        # Thank you responses
        thanks = {
            'thanks': f"{voice_indicator}You're welcome! 😊 Anything else I can help with?",
            'thank you': f"{voice_indicator}You're very welcome! 😊 Happy to help!",
            'appreciate': f"{voice_indicator}I appreciate that! 😊 Let me know if you need more help."
        }
        
        # Status check requests
        if any(word in text_lower for word in ['status', 'ticket', 'check', 'my']):
            await update.message.reply_text(
                f"{voice_indicator}Let me check your tickets! Use /status to see all your active tickets. 📊"
            )
            return
        
        # Help requests
        if any(word in text_lower for word in ['help', 'how', 'what can you do', 'commands']):
            await self.help_command(update, None)
            return
        
        # Check for specific responses
        for keyword, response in {**greetings, **thanks}.items():
            if keyword in text_lower:
                await update.message.reply_text(response)
                return
        
        # Default comprehensive response
        default_response = f"""{voice_indicator}*I'm here to help!* Here's what I can do:

🔧 *Report Issues:*
• "My AC is broken"
• "There's a leak in room 304"
• "WiFi is down"

🎉 *Events & Polls:*
• "What events are coming up?"
• "Create a poll about movie night"
• "Should we have a barbecue?"

📊 *Check Status:*
• /status - Your tickets
• /events - Upcoming events
• /activepolls - Current polls

🎤 *Voice Commands:*
• Send voice messages anytime!
• I understand natural speech

💡 *Need Help?*
• /help - Full command list
• Just describe what you need!

What would you like to do? 😊"""
        
        await update.message.reply_text(default_response, parse_mode='Markdown')

    # COMMAND HANDLERS
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced help command"""
        voice_status = "✅ Available" if self.voice_enabled else "❌ Disabled"
        
        help_text = f"""🤖 *HomeHive Buddy - Complete Guide*

🎤 *Voice Assistant:* {voice_status}

*🔧 Maintenance & Issues:*
• Report problems naturally: "My AC is broken"
• Include location: "Leak in apartment 2B"
• Use voice messages for quick reporting
• /status - Check your tickets

*🎉 Events & Community:*
• /events - View upcoming events
• "What events are coming up?"
• Create suggestions naturally

*🗳️ Polls & Voting:*
• /createpoll <question> - Create a poll
• /activepolls - View active polls
• /pollresults <poll_id> - View results
• Say: "Create a poll about movie night"

*💬 Natural Language:*
✅ "My WiFi isn't working"
✅ "Schedule maintenance check"
✅ "Should we have game night?"
✅ Send voice messages anytime!

*🎤 Voice Tips:*
• Speak clearly and at normal pace
• Reduce background noise
• Hold device 6 inches from mouth
• Try again if not understood

*Quick Commands:*
/start - Welcome message
/help - This help menu
/status - Your ticket status

Just talk to me naturally - I understand! 😊"""
        
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user's ticket status"""
        try:
            user_id = update.effective_user.id
            tickets = await self.get_user_tickets(user_id)
            
            if not tickets:
                await update.message.reply_text(
                    "📋 *No Active Tickets*\n\n"
                    "You don't have any tickets right now.\n\n"
                    "💡 *Need help?* Just describe your issue:\n"
                    "• \"My AC is not working\"\n"
                    "• \"There's a leak in my bathroom\"\n"
                    "• Send a voice message! 🎤",
                    parse_mode='Markdown'
                )
                return
            
            response = f"📊 *Your Tickets ({len(tickets)} active):*\n\n"
            
            for i, ticket in enumerate(tickets[:5], 1):
                status_emoji = {
                    'Resolved': "✅",
                    'In Progress': "🔄", 
                    'Open': "🆕",
                    'Cancelled': "❌"
                }.get(ticket['status'], "❓")
                
                voice_indicator = "🎤" if ticket.get('is_voice_request') else "💬"
                
                response += f"{i}. {status_emoji} *{ticket['ticket_id']}* {voice_indicator}\n"
                response += f"   📝 {ticket['description'][:60]}{'...' if len(ticket['description']) > 60 else ''}\n"
                response += f"   📍 {ticket['location']} | 🚨 {ticket['priority']}\n"
                response += f"   📅 {ticket['created_at'][:16]}\n\n"
            
            if len(tickets) > 5:
                response += f"... and {len(tickets) - 5} more tickets\n\n"
            
            response += "💡 *Need to report another issue?* Just describe it or send a voice message!"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error fetching your tickets. Please try again.")

    async def events_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show upcoming events"""
        await self.show_upcoming_events(update)

    async def maintenance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show maintenance info and quick actions"""
        maintenance_text = """🔧 *Maintenance Dashboard*

*🚨 Quick Issue Reporting:*
• Just describe your problem naturally
• "My AC is not working"
• "There's a leak in room 304"
• "WiFi is down in my unit"
• 🎤 Send voice messages for instant reporting!

*📊 Your Status:*
• /status - Check your active tickets
• Get real-time updates on repairs

*⏰ Maintenance Schedule:*
• 🔄 Regular inspections: Every Monday
• 🧹 Deep cleaning: First Sunday of month  
• ⚙️ Equipment check: Bi-weekly
• 🌡️ HVAC maintenance: Quarterly

*🚨 Emergency Contacts:*
• Fire/Medical: 911
• Building Emergency: Contact management
• After hours urgent issues: [Contact info]

*💡 Tips:*
• Be specific about location
• Mention urgency level
• Include photos if helpful
• Use voice messages for quick reports

Ready to help! Just tell me what's wrong. 😊"""
        
        # Add quick action buttons
        keyboard = [
            [InlineKeyboardButton("🔧 Report Issue", callback_data="report_issue")],
            [InlineKeyboardButton("📊 My Tickets", callback_data="show_status")]
        ]
        
        await update.message.reply_text(
            maintenance_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # POLL COMMANDS
    async def create_event_poll_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /createpoll command"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "🗳️ *Create a Poll*\n\n"
                    "*Usage:* /createpoll <your question>\n\n"
                    "*Examples:*\n"
                    "• /createpoll Should we have movie night?\n"
                    "• /createpoll What time for the barbecue?\n"
                    "• /createpoll Pool party this weekend?\n\n"
                    "*Or just say naturally:*\n"
                    "• \"Create a poll about game night\"\n"
                    "• \"Should we have a barbecue?\"\n"
                    "• 🎤 Send a voice message!",
                    parse_mode='Markdown'
                )
                return

            question = ' '.join(context.args)
            await self.create_poll_from_text(update, question)

        except Exception as e:
            logger.error(f"Error in createpoll command: {e}")
            await update.message.reply_text("❌ Error creating poll. Please try again.")

    async def show_active_polls(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active polls"""
        try:
            active_polls = self.event_db.get_active_polls()
            
            if not active_polls:
                await update.message.reply_text(
                    "🗳️ *No Active Polls*\n\n"
                    "There are no polls running right now.\n\n"
                    "💡 *Want to create one?*\n"
                    "• /createpoll <question>\n"
                    "• Say: \"Create a poll about...\"\n"
                    "• 🎤 Send a voice message!\n\n"
                    "Example: \"Should we have movie night?\"",
                    parse_mode='Markdown'
                )
                return
            
            response = f"🗳️ *Active Polls ({len(active_polls)}):*\n\n"
            
            for i, poll in enumerate(active_polls[:5], 1):
                response += f"{i}. 📊 *{poll['poll_id']}*\n"
                response += f"   ❓ {poll['question']}\n"
                response += f"   👥 {poll.get('total_votes', 0)} votes\n"
                response += f"   👤 By: {poll.get('creator_name', 'Unknown')}\n"
                try:
                    created = datetime.fromisoformat(poll['created_at'])
                    response += f"   📅 {created.strftime('%m/%d %H:%M')}\n\n"
                except:
                    response += f"   📅 Recently created\n\n"
            
            if len(active_polls) > 5:
                response += f"... and {len(active_polls) - 5} more polls\n\n"
            
            response += "💡 Use /pollresults <poll_id> for detailed results!"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing active polls: {e}")
            await update.message.reply_text("❌ Error fetching polls. Please try again.")

    async def show_poll_results(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed poll results"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "📊 *Poll Results*\n\n"
                    "*Usage:* /pollresults <poll_id>\n\n"
                    "*Example:* /pollresults POLL1001\n\n"
                    "💡 Use /activepolls to see available poll IDs",
                    parse_mode='Markdown'
                )
                return

            poll_id = context.args[0].upper()
            poll_data = self.event_db.get_poll(poll_id)
            
            if not poll_data:
                await update.message.reply_text(
                    f"❌ *Poll Not Found*\n\n"
                    f"Poll `{poll_id}` doesn't exist.\n\n"
                    f"💡 Use /activepolls to see available polls",
                    parse_mode='Markdown'
                )
                return
            
            results_text = f"📊 *Poll Results: {poll_data['poll_id']}*\n\n"
            results_text += f"❓ *Question:* {poll_data['question']}\n"
            results_text += f"👤 *Created by:* {poll_data.get('creator_name', 'Unknown')}\n"
            results_text += f"👥 *Total Votes:* {poll_data.get('total_votes', 0)}\n\n"
            
            votes = poll_data.get('votes', {})
            total_votes = poll_data.get('total_votes', 0)
            
            if total_votes > 0:
                results_text += "*📈 Results:*\n"
                for option, voters in votes.items():
                    vote_count = len(voters) if isinstance(voters, list) else 0
                    percentage = (vote_count / total_votes * 100) if total_votes > 0 else 0
                    
                    # Create visual bar
                    filled_bars = int(percentage / 10)
                    empty_bars = 10 - filled_bars
                    bar = "█" * filled_bars + "░" * empty_bars
                    
                    results_text += f"\n• *{option}*\n"
                    results_text += f"  {vote_count} votes ({percentage:.1f}%)\n"
                    results_text += f"  {bar}\n"
            else:
                results_text += "\n*No votes yet!* 🗳️\n"
                results_text += "Be the first to vote on this poll!"
            
            await update.message.reply_text(results_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing poll results: {e}")
            await update.message.reply_text("❌ Error fetching poll results. Please try again.")

    # CALLBACK HANDLERS
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith("status_"):
                ticket_id = data.replace("status_", "")
                await self.show_ticket_status_callback(query, ticket_id)
            elif data.startswith("cancel_"):
                ticket_id = data.replace("cancel_", "")
                await self.cancel_ticket_callback(query, ticket_id)
            elif data == "show_events":
                await self.show_upcoming_events_callback(query)
            elif data == "show_polls":
                await self.show_active_polls_callback(query)
            elif data == "create_poll_prompt":
                await self.create_poll_prompt_callback(query)
            elif data == "report_issue":
                await self.report_issue_callback(query)
            elif data == "show_status":
                await self.show_status_callback(query)
            else:
                await query.edit_message_text("❓ Unknown action.")
                
        except Exception as e:
            logger.error(f"Error handling callback: {e}")
            try:
                await query.edit_message_text("❌ Error processing request.")
            except:
                pass

    async def show_ticket_status_callback(self, query, ticket_id: str):
        """Show ticket status via callback"""
        try:
            cursor = self.ticket_db.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [col[0] for col in cursor.description]
                ticket = dict(zip(columns, result))
                
                voice_indicator = "🎤 (Voice)" if ticket.get('is_voice_request') else "💬 (Text)"
                
                response = f"""📊 *Ticket {ticket_id} Details* {voice_indicator}

📝 *Description:* {ticket['description']}
📍 *Location:* {ticket['location']}
🚨 *Priority:* {ticket['priority']}
⏰ *Status:* {ticket['status']}
📅 *Created:* {ticket['created_at'][:16]}

{self._get_eta_message(ticket['priority'])}"""
            else:
                response = f"❌ Ticket {ticket_id} not found."
            
            await query.edit_message_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing ticket status: {e}")
            await query.edit_message_text("❌ Error fetching ticket details.")

    async def cancel_ticket_callback(self, query, ticket_id: str):
        """Cancel a ticket via callback"""
        try:
            # Update ticket status
            self.ticket_db.execute(
                "UPDATE tickets SET status = 'Cancelled' WHERE ticket_id = ?",
                (ticket_id,),
                commit=True
            )
            
            response = f"""❌ *Ticket {ticket_id} Cancelled*

Your maintenance request has been cancelled successfully.

💡 *Need help with something else?*
• Describe your issue naturally
• Send a voice message 🎤
• Use /help for more options

Ready to assist you anytime! 😊"""
            
            await query.edit_message_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error cancelling ticket: {e}")
            await query.edit_message_text("❌ Error cancelling ticket.")

    async def show_upcoming_events_callback(self, query):
        """Show upcoming events via callback"""
        try:
            events = self.event_db.get_events(status='active')
            
            if not events:
                response = """📅 *No Upcoming Events*

No events scheduled right now.

💡 *Want to organize something?*
• Create a poll to gauge interest
• Say: "Should we have movie night?"
• Use /createpoll command"""
            else:
                response = "📅 *Upcoming Events:*\n\n"
                for i, event in enumerate(events[:3], 1):
                    response += f"{i}. 🎉 *{event['name']}*\n"
                    if event.get('description'):
                        response += f"   📝 {event['description']}\n"
                    response += "\n"
                
                response += "💡 Use /createpoll to suggest new events!"
            
            await query.edit_message_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing events: {e}")
            await query.edit_message_text("❌ Error fetching events.")

    async def show_active_polls_callback(self, query):
        """Show active polls via callback"""
        try:
            active_polls = self.event_db.get_active_polls()
            
            if not active_polls:
                response = """🗳️ *No Active Polls*

No polls running right now.

💡 *Create one:*
• /createpoll <question>
• Say: "Create a poll about..."
• 🎤 Send voice message!"""
            else:
                response = f"🗳️ *Active Polls ({len(active_polls)}):*\n\n"
                for i, poll in enumerate(active_polls[:3], 1):
                    response += f"{i}. 📊 {poll['poll_id']}\n"
                    response += f"   ❓ {poll['question'][:50]}{'...' if len(poll['question']) > 50 else ''}\n"
                    response += f"   👥 {poll.get('total_votes', 0)} votes\n\n"
                
                response += "Use /pollresults <poll_id> for details!"
            
            await query.edit_message_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing polls: {e}")
            await query.edit_message_text("❌ Error fetching polls.")

    async def create_poll_prompt_callback(self, query):
        """Prompt user to create a poll"""
        response = """🗳️ *Create a New Poll*

*Ways to create a poll:*

1️⃣ *Command:* /createpoll <question>
   Example: `/createpoll Should we have movie night?`

2️⃣ *Natural language:*
   Just say: "Create a poll about game night"

3️⃣ *Voice message:* 🎤
   Record: "Should we organize a barbecue?"

*Popular poll topics:*
• Community events
• Amenity usage schedules  
• Building improvements
• Social activities

Ready to create your poll? 😊"""
        
        await query.edit_message_text(response, parse_mode='Markdown')

    async def report_issue_callback(self, query):
        """Prompt user to report an issue"""
        response = """🔧 *Report Maintenance Issue*

*How to report:*

1️⃣ *Describe naturally:*
   "My AC is not working"
   "There's a leak in room 2B"

2️⃣ *Voice message:* 🎤
   Record your issue for instant reporting

3️⃣ *Be specific:*
   • What's the problem?
   • Where is it located?
   • How urgent is it?

*Examples:*
• "WiFi is down in apartment 304"
• "Broken light in hallway, 2nd floor"
• "Elevator making strange noises"

Just describe your issue - I'll create a ticket! 🎫"""
        
        await query.edit_message_text(response, parse_mode='Markdown')

    async def show_status_callback(self, query):
        """Show user status via callback"""
        try:
            # Get user ID from query
            user_id = query.from_user.id
            tickets = await self.get_user_tickets(user_id)
            
            if not tickets:
                response = """📋 *No Active Tickets*

You don't have any tickets right now.

💡 *Need maintenance help?*
• Describe your issue
• Send voice message 🎤
• I'll create a ticket instantly!"""
            else:
                response = f"📊 *Your Tickets ({len(tickets)}):*\n\n"
                for i, ticket in enumerate(tickets[:3], 1):
                    status_emoji = {"Resolved": "✅", "In Progress": "🔄", "Open": "🆕"}.get(ticket['status'], "❓")
                    voice_indicator = "🎤" if ticket.get('is_voice_request') else "💬"
                    
                    response += f"{i}. {status_emoji} *{ticket['ticket_id']}* {voice_indicator}\n"
                    response += f"   📝 {ticket['description'][:40]}...\n"
                    response += f"   📅 {ticket['created_at'][:10]}\n\n"
                
                if len(tickets) > 3:
                    response += f"... and {len(tickets) - 3} more\n\n"
                
                response += "Use /status for full details!"
            
            await query.edit_message_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing status: {e}")
            await query.edit_message_text("❌ Error fetching your tickets.")

    # POLL HANDLERS
    async def handle_poll_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle poll answers and update database"""
        try:
            poll_answer = update.poll_answer
            user = poll_answer.user
            telegram_poll_id = poll_answer.poll_id
            option_ids = poll_answer.option_ids
            
            # Find poll in database
            cursor = self.event_db.conn.cursor()
            cursor.execute('SELECT * FROM polls WHERE telegram_poll_id = ?', (telegram_poll_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Poll not found for telegram_poll_id: {telegram_poll_id}")
                return
            
            # Get poll data
            columns = [col[0] for col in cursor.description]
            poll_data = dict(zip(columns, result))
            poll_data['options'] = json.loads(poll_data['options'])
            poll_data['votes'] = json.loads(poll_data['votes']) if poll_data['votes'] else {}
            
            # Prepare user vote data
            user_vote_data = {
                'user_id': user.id,
                'username': user.username or user.first_name or "Unknown",
                'voted_at': datetime.now().isoformat(),
                'option_ids': option_ids
            }
            
            votes = poll_data.get('votes', {})
            options = poll_data.get('options', [])
            
            # Remove user's previous votes
            for option in options:
                if option in votes and isinstance(votes[option], list):
                    votes[option] = [v for v in votes[option] if v.get('user_id') != user.id]
            
            # Add new votes
            for option_id in option_ids:
                if option_id < len(options):
                    option = options[option_id]
                    if option not in votes:
                        votes[option] = []
                    votes[option].append(user_vote_data)
            
            # Calculate total votes
            total_votes = sum(len(voters) for voters in votes.values() if isinstance(voters, list))
            
            # Update database
            cursor.execute('''
                UPDATE polls 
                SET votes = ?, total_votes = ?
                WHERE poll_id = ?
            ''', (json.dumps(votes), total_votes, poll_data['poll_id']))
            self.event_db.conn.commit()
            
            logger.info(f"Poll vote recorded: User {user.id} voted in poll {poll_data['poll_id']}")
            
        except Exception as e:
            logger.error(f"Error handling poll answer: {e}")

    # DATABASE HELPER METHODS
    async def log_user(self, user):
        """Log user information"""
        try:
            query = """
            INSERT OR REPLACE INTO users (user_id, username, first_name, last_name, last_seen)
            VALUES (?, ?, ?, ?, ?)
            """
            params = (
                user.id,
                user.username,
                user.first_name,
                user.last_name,
                datetime.now().isoformat()
            )
            
            self.ticket_db.execute(query, params, commit=True)
            
        except Exception as e:
            logger.error(f"Error logging user: {e}")

    async def save_ticket(self, ticket_data):
        """Save ticket to database"""
        try:
            query = """
            INSERT INTO tickets (
                ticket_id, user_id, username, description, location, 
                priority, status, created_at, is_voice_request
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                ticket_data['ticket_id'],
                ticket_data['user_id'],
                ticket_data['username'],
                ticket_data['description'],
                ticket_data['location'],
                ticket_data['priority'],
                ticket_data['status'],
                ticket_data['created_at'],
                int(ticket_data['is_voice_request'])
            )
            
            self.ticket_db.execute(query, params, commit=True)
            logger.info(f"Ticket saved: {ticket_data['ticket_id']}")
            
        except Exception as e:
            logger.error(f"Error saving ticket: {e}")
            raise

    async def get_user_tickets(self, user_id: int) -> List[Dict]:
        """Get user's tickets from database"""
        try:
            query = "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC"
            cursor = self.ticket_db.execute(query, (user_id,))
            
            columns = [col[0] for col in cursor.description]
            tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            logger.info(f"Retrieved {len(tickets)} tickets for user {user_id}")
            return tickets
            
        except Exception as e:
            logger.error(f"Error fetching tickets for user {user_id}: {e}")
            return []

    # CLEANUP AND SHUTDOWN
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'ticket_db') and self.ticket_db:
                self.ticket_db.close()
            if hasattr(self, 'event_db') and self.event_db:
                self.event_db.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def run(self):
        """Run the bot with proper error handling"""
        print("🤖 HomeHive Voice Assistant starting up...")
        
        # Configuration check
        print("\n=== Configuration Check ===")
        print(f"TELEGRAM_TOKEN: {'✅ Set' if self.bot_token else '❌ Missing'}")
        print(f"Database: ✅ SQLite (Tickets & Events)")
        print(f"AI Features: {'✅ Gemini Available' if self.gemini_model else '⚠️ Basic Mode'}")
        print(f"Voice Processing: {'✅ Enabled' if self.voice_enabled else '❌ Disabled'}")
        
        if not self.bot_token:
            print("❌ TELEGRAM_TOKEN is required!")
            print("Please set it in your .env file")
            return
        
        if not self.voice_enabled:
            print("\n⚠️ Voice features disabled. To enable:")
            print("pip install SpeechRecognition pydub requests")
        
        try:
            print("\n🚀 Starting bot...")
            print("Press Ctrl+C to stop")
            
            self.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=['message', 'callback_query', 'poll', 'poll_answer'],
                close_loop=False
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            print(f"\n❌ Bot error: {str(e)}")
        finally:
            print("🧹 Cleaning up...")
            self.cleanup()

# MAIN EXECUTION
if __name__ == "__main__":
    print("🏠 HomeHive Voice Assistant Bot")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    
    if not telegram_token:
        print("❌ TELEGRAM_TOKEN environment variable is required!")
        print("\n📝 Setup Instructions:")
        print("1. Create a .env file in the same directory")
        print("2. Add: TELEGRAM_TOKEN=your_bot_token_here")
        print("3. Optional: Add GEMINI_API_KEY=your_gemini_key for AI features")
        print("\n🤖 Get a bot token from @BotFather on Telegram")
        exit(1)
    
    # Check dependencies
    missing_deps = []
    
    if not VOICE_AVAILABLE:
        missing_deps.extend(['SpeechRecognition', 'pydub', 'requests'])
    
    if missing_deps:
        print(f"\n⚠️ Missing optional dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("Voice features will be disabled without these packages.\n")
    
    try:
        # Create and run bot
        print("🔧 Initializing bot...")
        bot = HomeHiveBuddy(telegram_token=telegram_token)
        
        print("✅ Bot initialized successfully!")
        bot.run()
        
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Verify TELEGRAM_TOKEN in .env file")
        print("2. Check internet connection")
        print("3. Install required packages:")
        print("   pip install python-telegram-bot python-dotenv")
        print("4. For voice features:")
        print("   pip install SpeechRecognition pydub requests")
        
        import traceback
        print(f"\n📋 Full error details:")
        traceback.print_exc()
        exit(1)