from src.interfaces.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

class SQLAlchemyChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_maker, session_id: str, default_customer_id="CUST103161"):
        self.session_maker = session_maker
        self.session_id = session_id
        self.default_customer_id = default_customer_id

    @property
    def messages(self) -> list[BaseMessage]:
        with self.session_maker() as session:
            from src.database.models import ChatMessage
            msgs = session.query(ChatMessage)\
                .filter_by(session_id=self.session_id)\
                .order_by(ChatMessage.timestamp.asc())\
                .all()

            langchain_msgs = []
            for m in msgs:
                if m.role == "user":
                    langchain_msgs.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    langchain_msgs.append(AIMessage(content=m.content))
                else:
                    langchain_msgs.append(SystemMessage(content=m.content))
            return langchain_msgs

    def add_message(self, message: BaseMessage) -> None:
        with self.session_maker() as session:
            from src.database.models import ChatSession, ChatMessage
            # Ensure chat session exists
            chat_session = session.query(ChatSession).filter_by(session_id=self.session_id).first()
            if not chat_session:
                chat_session = ChatSession(session_id=self.session_id, customer_id=self.default_customer_id)
                session.add(chat_session)
                session.commit()

            role = "user"
            if isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"

            msg = ChatMessage(session_id=self.session_id, role=role, content=message.content)
            session.add(msg)
            session.commit()

    def clear(self) -> None:
        with self.session_maker() as session:
            from src.database.models import ChatSession, ChatMessage
            session.query(ChatMessage).filter_by(session_id=self.session_id).delete()
            session.query(ChatSession).filter_by(session_id=self.session_id).delete()
            session.commit()

class SQLAlchemyMemory(BaseMemory):
    def __init__(self, session_maker, default_customer_id="CUST103161"):
        """
        Memory manager backed by SQLAlchemy database and wrapped as LangChain BaseChatMessageHistory.
        """
        self.session_maker = session_maker
        self.default_customer_id = default_customer_id

    def _get_history_obj(self, session_id: str) -> SQLAlchemyChatMessageHistory:
        return SQLAlchemyChatMessageHistory(self.session_maker, session_id, self.default_customer_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self._get_history_obj(session_id)
        if role == "user":
            history.add_message(HumanMessage(content=content))
        elif role == "assistant":
            history.add_message(AIMessage(content=content))
        else:
            history.add_message(SystemMessage(content=content))

    def get_history(self, session_id: str, limit: int = 10, max_tokens: int = None) -> list[dict]:
        history = self._get_history_obj(session_id)
        msgs = history.messages
        recent = msgs[-limit:] if len(msgs) > limit else msgs
        formatted = []
        for m in recent:
            role = "user"
            if isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            formatted.append({'role': role, 'content': m.content})

        if max_tokens is not None and max_tokens > 0:
            from src.utils.token_budget import budget_chat_history
            formatted, _, _ = budget_chat_history(formatted, max_tokens=max_tokens)

        return formatted

    def clear(self, session_id: str) -> None:
        self._get_history_obj(session_id).clear()

class StreamlitSessionMemory(BaseMemory):
    def __init__(self):
        """Volatile in-memory chat storage utilizing LangChain ChatMessageHistory."""
        from langchain_community.chat_message_histories import ChatMessageHistory
        self.sessions = {}
        self.chat_history_class = ChatMessageHistory

    def _get_history_obj(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = self.chat_history_class()
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self._get_history_obj(session_id)
        if role == "user":
            history.add_message(HumanMessage(content=content))
        elif role == "assistant":
            history.add_message(AIMessage(content=content))
        else:
            history.add_message(SystemMessage(content=content))

    def get_history(self, session_id: str, limit: int = 10, max_tokens: int = None) -> list[dict]:
        history = self._get_history_obj(session_id)
        msgs = history.messages
        recent = msgs[-limit:] if len(msgs) > limit else msgs
        formatted = []
        for m in recent:
            role = "user"
            if isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            formatted.append({'role': role, 'content': m.content})

        if max_tokens is not None and max_tokens > 0:
            from src.utils.token_budget import budget_chat_history
            formatted, _, _ = budget_chat_history(formatted, max_tokens=max_tokens)

        return formatted

    def clear(self, session_id: str) -> None:
        self._get_history_obj(session_id).clear()

