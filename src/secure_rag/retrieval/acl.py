from secure_rag.models import Chunk, User


def filter_authorized_chunks(user: User, chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if user.clearance >= chunk.clearance]
