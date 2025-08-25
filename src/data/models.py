"""
Database models for chemical inventory system.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class Chemical(Base):
    """Chemical inventory record model."""
    
    __tablename__ = 'chemicals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic chemical information
    name = Column(String(255), nullable=False)
    cas_number = Column(String(50))
    formula = Column(String(100))
    molecular_weight = Column(String(50))
    
    # Product information
    concentration = Column(String(100))
    manufacturer = Column(String(255))
    lot_number = Column(String(100))
    catalog_number = Column(String(100))
    expiry_date = Column(DateTime)
    
    # Safety and storage
    hazard_symbols = Column(JSON)  # Store as JSON array
    hazard_statements = Column(Text)
    storage_conditions = Column(Text)
    
    # Image information
    image_path = Column(String(500), nullable=False)
    image_filename = Column(String(255))
    image_size = Column(Integer)
    
    # OCR and extraction metadata
    ocr_text = Column(Text)  # Raw OCR output
    confidence_score = Column(Integer)  # OCR confidence (0-100)
    extraction_method = Column(String(50))  # 'ocr', 'ai_enhanced', etc.
    
    # Database tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Chemical(name='{self.name}', cas='{self.cas_number}')>"
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            'id': str(self.id),
            'name': self.name,
            'cas_number': self.cas_number,
            'formula': self.formula,
            'molecular_weight': self.molecular_weight,
            'concentration': self.concentration,
            'manufacturer': self.manufacturer,
            'lot_number': self.lot_number,
            'catalog_number': self.catalog_number,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'hazard_symbols': self.hazard_symbols,
            'hazard_statements': self.hazard_statements,
            'storage_conditions': self.storage_conditions,
            'image_filename': self.image_filename,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }