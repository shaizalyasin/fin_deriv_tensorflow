from enum import Enum


class PutCall(str, Enum):
    PUT: str = 'PUT'
    CALL: str = 'CALL'
