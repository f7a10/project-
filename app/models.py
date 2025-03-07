import os
import uuid
import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import event
from flask import current_app

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # Relationships
    uploads = db.relationship('DataUpload', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class DataUpload(db.Model):
    """Model to represent a data upload session."""
    __tablename__ = 'data_uploads'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_id = db.Column(db.String(36), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    file_count = db.Column(db.Integer, default=0)
    processed = db.Column(db.Boolean, default=False)

    # Relationships
    files = db.relationship('UploadedFile', backref='upload', lazy=True, cascade="all, delete-orphan")
    analyses = db.relationship('DataAnalysis', backref='upload', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<DataUpload {self.id}>'


class UploadedFile(db.Model):
    """Model to store information about uploaded files."""
    __tablename__ = 'uploaded_files'

    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.String(36), db.ForeignKey('data_uploads.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(50))
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<UploadedFile {self.filename}>'

    @property
    def full_path(self):
        """Return the full path to the file."""
        return os.path.join(current_app.config['UPLOAD_FOLDER'], self.file_path)


class DataAnalysis(db.Model):
    """Model to store data analysis results."""
    __tablename__ = 'data_analyses'

    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.String(36), db.ForeignKey('data_uploads.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)
    results = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # Relationships
    visualizations = db.relationship('Visualization', backref='analysis', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<DataAnalysis {self.id} - {self.analysis_type}>'


class Visualization(db.Model):
    """Model to store visualization data."""
    __tablename__ = 'visualizations'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('data_analyses.id'), nullable=False)
    chart_type = db.Column(db.String(50), nullable=False)
    chart_data = db.Column(JSON)
    title = db.Column(db.String(255))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<Visualization {self.id} - {self.chart_type}>'


class ChatMessage(db.Model):
    """Model to store chat messages between user and AI."""
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)  # True if user message, False if AI message
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<ChatMessage {self.id}>'


# Event listeners for cleanup
@event.listens_for(DataUpload, 'after_delete')
def delete_upload_files(mapper, connection, target):
    """Delete files from disk when DataUpload is deleted."""
    for file in target.files:
        try:
            if os.path.exists(file.full_path):
                os.remove(file.full_path)
        except Exception as e:
            # Log the error but don't stop the deletion process
            print(f"Error deleting file {file.full_path}: {e}")


# Helper functions
def init_db():
    """Initialize the database by creating all tables."""
    db.create_all()


def get_or_create_session(session_id=None):
    """Get or create a new upload session."""
    if not session_id:
        session_id = str(uuid.uuid4())

    upload = DataUpload(
        session_id=session_id
    )
    db.session.add(upload)
    db.session.commit()

    return upload