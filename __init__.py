from flask import Flask
from pymongo import MongoClient
from flask_mail import Mail
from flask_socketio import SocketIO
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "hamckjoker254"
app.config["MONGO_URI"] = "mongodb://localhost:27017/mkulimahub"
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # e.g., smtp.gmail.com
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'hamckjoker@gmail.com'
app.config['MAIL_PASSWORD'] = 'twpm kdit glif eibx'
app.config['MAIL_DEFAULT_SENDER'] = 'hamckjoker@gmail.com'


base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['AI_FOLDER'] = os.path.join(base_dir, 'static', 'AI_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AI_FOLDER'], exist_ok=True)


socketio = SocketIO(app)
mail = Mail(app)

client = MongoClient(app.config["MONGO_URI"])
db = client.get_database()
Fuser_collection = db.Fusers
Buser_collection = db.Busers
posts = db.posts
password_reset_token = db.password_reset_tokens
messages = db.messages
AI_collection=db.AI

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from .auth import auth as auth_bp
from .main import main as main_bp
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)


