from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


# User Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    display_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    display_name: str
    bio: Optional[str] = None
    profile_picture: Optional[str] = None
    average_rating: float = 0.0
    total_ratings: int = 0
    created_at: datetime
    is_friend: Optional[bool] = False
    friend_request_status: Optional[str] = None  # pending, accepted, none
    banner: Optional[str] = None  # top-rated, try-harder, None
    banner_expiry: Optional[datetime] = None
    is_verified: bool = False

class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    bio: Optional[str] = None
    profile_picture: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Rating Models
class RatingCreate(BaseModel):
    stars: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class RatingResponse(BaseModel):
    id: str
    rated_user_id: str
    rater_user_id: str
    rater_username: str
    rater_display_name: str
    rater_profile_picture: Optional[str] = None
    stars: int
    comment: Optional[str] = None
    created_at: datetime

# Friend Models
class FriendRequest(BaseModel):
    friend_id: str

class FriendResponse(BaseModel):
    id: str
    user_id: str
    friend_id: str
    friend_username: str
    friend_display_name: str
    friend_profile_picture: Optional[str] = None
    status: str  # pending, accepted, rejected
    created_at: datetime

# Competition Models
class CompetitionCreate(BaseModel):
    name: str

class CompetitionResponse(BaseModel):
    id: str
    name: str
    start_date: datetime
    end_date: datetime
    status: str  # active, completed
    participants: List[dict]
    winner_id: Optional[str] = None
    loser_id: Optional[str] = None
    created_by: str

class CompetitionJoin(BaseModel):
    competition_id: str

# Group Models
class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None

class GroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    average_rating: float = 0.0
    member_count: int = 0
    members: List[str] = []
    created_by: str
    created_at: datetime

class GroupMemberInvite(BaseModel):
    user_id: str

# Session Models
class SessionData(BaseModel):
    session_id: str

class EmergentUserData(BaseModel):
    id: str
    email: str
    name: str
    picture: str
    session_token: str

# Email Verification Models
class VerifyEmailRequest(BaseModel):
    code: str

class ResendCodeRequest(BaseModel):
    email: EmailStr

class VerificationResponse(BaseModel):
    message: str
    is_verified: bool
