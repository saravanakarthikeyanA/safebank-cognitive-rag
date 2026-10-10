import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class SupportTicket(Base):
    __tablename__ = 'support_tickets'

    ticket_id = Column(String(50), primary_key=True)
    date_created = Column(String(50))
    customer_id = Column(String(50), index=True)
    channel = Column(String(50))
    category = Column(String(50))
    sub_category = Column(String(100))
    query_text = Column(Text)
    sentiment = Column(String(50))
    risk_level = Column(String(50))
    resolution_text = Column(Text)
    resolution_time_minutes = Column(Integer)
    resolved_by = Column(String(100))
    customer_satisfaction = Column(Integer, nullable=True)
    escalated = Column(Boolean)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Transaction(Base):
    __tablename__ = 'transactions'

    transaction_id = Column(String(50), primary_key=True)
    account_id = Column(String(50), index=True)
    timestamp = Column(String(50))
    amount_inr = Column(Float)
    merchant_name = Column(String(100))
    merchant_category = Column(String(100))
    transaction_type = Column(String(50))
    city = Column(String(100))
    hour_of_day = Column(Integer)
    day_of_week = Column(String(50))
    is_international = Column(Boolean)
    velocity_flag = Column(Boolean)
    geo_anomaly_flag = Column(Boolean)
    high_amount_flag = Column(Boolean)
    fraud_label = Column(Integer)  # 1 = Fraud, 0 = Legitimate
    fraud_reason = Column(String(200), nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    session_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey('chat_sessions.session_id'), index=True)
    role = Column(String(50))  # 'user', 'assistant', 'system'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class PipelineLog(Base):
    __tablename__ = 'pipeline_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    query = Column(Text)
    intent = Column(String(50))
    sentiment = Column(String(50))
    fraud_risk = Column(String(50))
    suggested_action = Column(String(100))
    response = Column(Text)
    latency_ms = Column(Integer)
    trace_json = Column(Text)  # Serialized trace execution steps
