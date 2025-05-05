# core/hashing.py
from passlib.context import CryptContext

# Configura bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Hasher:
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Genera un hash bcrypt de la contraseña.
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verifica que `plain_password` coincida con `hashed_password`.
        """
        return pwd_context.verify(plain_password, hashed_password)
