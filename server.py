from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request, Response, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import httpx
import qrcode
import io
import base64
import resend
import random
import string

from models import (
    UserCreate, UserLogin, UserResponse, UserUpdate, Token,
    RatingCreate, RatingResponse,
    FriendRequest, FriendResponse,
    CompetitionCreate, CompetitionResponse, CompetitionJoin,
    GroupCreate, GroupResponse, GroupMemberInvite,
    SessionData, EmergentUserData,
    VerifyEmailRequest, ResendCodeRequest, VerificationResponse
)
from auth import (
    get_password_hash, verify_password, create_access_token, 
    get_current_user, get_current_user_optional
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Resend configuration for email
resend.api_key = "re_EDuB3JG7_293dQz9dvJo5WGvCEXFVKqrA"

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Helper functions
def user_to_response(user: dict, current_user_id: Optional[str] = None) -> UserResponse:
    """Convert database user to UserResponse"""
    # Check if banner is expired and clear it
    banner = user.get("banner")
    banner_expiry = user.get("banner_expiry")
    
    if banner and banner_expiry:
        if banner_expiry < datetime.now(timezone.utc):
            banner = None
            banner_expiry = None
    
    response = UserResponse(
        id=str(user["_id"]),
        username=user["username"],
        email=user["email"],
        display_name=user["display_name"],
        bio=user.get("bio"),
        profile_picture=user.get("profile_picture"),
        average_rating=user.get("average_rating", 0.0),
        total_ratings=user.get("total_ratings", 0),
        created_at=user["created_at"],
        banner=banner,
        banner_expiry=banner_expiry,
        is_verified=user.get("is_verified", False)
    )
    
    # Check friendship status if current_user_id is provided
    if current_user_id and str(user["_id"]) != current_user_id:
        # Will be implemented in friend routes
        response.is_friend = False
        response.friend_request_status = "none"
    
    return response


async def update_user_rating(user_id: str):
    """Recalculate and update user's average rating"""
    ratings = await db.ratings.find({"rated_user_id": user_id}).to_list(None)
    if ratings:
        avg_rating = sum(r["stars"] for r in ratings) / len(ratings)
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"average_rating": round(avg_rating, 2), "total_ratings": len(ratings)}}
        )


def generate_verification_code() -> str:
    """Generate a 6-digit verification code"""
    return ''.join(random.choices(string.digits, k=6))


async def send_verification_email(email: str, code: str, username: str):
    """Send verification code via Resend"""
    try:
        params = {
            "from": "Rate Me <onboarding@resend.dev>",
            "to": [email],
            "subject": "Verify Your Email - Rate Me",
            "html": f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #0891b2;">Welcome to Rate Me, {username}!</h2>
                <p>Thanks for signing up! Please verify your email address to access all features.</p>
                <div style="background-color: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 0; font-size: 14px; color: #666;">Your verification code is:</p>
                    <h1 style="margin: 10px 0; font-size: 36px; letter-spacing: 8px; color: #0891b2;">{code}</h1>
                </div>
                <p style="color: #666; font-size: 14px;">This code will expire in 15 minutes.</p>
                <p style="color: #666; font-size: 14px;">If you didn't request this, please ignore this email.</p>
            </div>
            """
        }
        
        resend.Emails.send(params)
        return True
    except Exception as e:
        logging.error(f"Error sending verification email: {str(e)}")
        return False


# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    """Register new user with username/password"""
    # Check if username exists
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    existing_email = await db.users.find_one({"email": user.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Generate verification code
    verification_code = generate_verification_code()
    code_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    # Create new user
    user_dict = {
        "username": user.username,
        "email": user.email,
        "password_hash": get_password_hash(user.password),
        "display_name": user.display_name,
        "bio": None,
        "profile_picture": None,
        "google_id": None,
        "average_rating": 0.0,
        "total_ratings": 0,
        "created_at": datetime.now(timezone.utc),
        "is_verified": False,
        "verification_code": verification_code,
        "verification_code_expiry": code_expiry
    }
    
    result = await db.users.insert_one(user_dict)
    user_dict["_id"] = result.inserted_id
    
    # Send verification email
    email_sent = await send_verification_email(user.email, verification_code, user.username)
    if not email_sent:
        logging.warning(f"Failed to send verification email to {user.email}")
    
    # Create access token
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_to_response(user_dict)
    )


@api_router.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """Login with username or email"""
    # Try to find user by username first, then email
    db_user = await db.users.find_one({
        "$or": [
            {"username": user.username},
            {"email": user.username}  # Allow email in username field
        ]
    })
    
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect username/email or password")
    
    access_token = create_access_token(data={"sub": str(db_user["_id"])})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_to_response(db_user)
    )


@api_router.post("/auth/google")
async def google_auth(session_data: SessionData, response: Response):
    """Process Google OAuth session from Emergent Auth"""
    try:
        # Call Emergent Auth API to get user data
        async with httpx.AsyncClient() as client:
            auth_response = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_data.session_id}
            )
            
            if auth_response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid session ID")
            
            user_data = EmergentUserData(**auth_response.json())
        
        # Check if user exists by email
        db_user = await db.users.find_one({"email": user_data.email})
        
        if not db_user:
            # Create new user from Google data
            username = user_data.email.split("@")[0] + "_" + user_data.id[:6]
            user_dict = {
                "username": username,
                "email": user_data.email,
                "password_hash": None,  # No password for OAuth users
                "display_name": user_data.name,
                "bio": None,
                "profile_picture": user_data.picture,
                "google_id": user_data.id,
                "average_rating": 0.0,
                "total_ratings": 0,
                "created_at": datetime.now(timezone.utc)
            }
            result = await db.users.insert_one(user_dict)
            user_dict["_id"] = result.inserted_id
            db_user = user_dict
        
        # Store session in database
        session_dict = {
            "user_id": str(db_user["_id"]),
            "session_token": user_data.session_token,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
            "created_at": datetime.now(timezone.utc)
        }
        await db.sessions.insert_one(session_dict)
        
        # Set httpOnly cookie
        response.set_cookie(
            key="session_token",
            value=user_data.session_token,
            httponly=True,
            secure=True,
            samesite="none",
            max_age=7 * 24 * 60 * 60,  # 7 days
            path="/"
        )
        
        # Also create JWT token for mobile app
        access_token = create_access_token(data={"sub": str(db_user["_id"])})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_to_response(db_user)
        )
    
    except Exception as e:
        logging.error(f"Google auth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(request: Request, current_user_id: str = Depends(get_current_user)):
    """Get current user info"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user_to_response(user)


# ==================== EMAIL VERIFICATION ROUTES ====================

@api_router.post("/auth/verify-email", response_model=VerificationResponse)
async def verify_email(request_data: VerifyEmailRequest, current_user_id: str = Depends(get_current_user)):
    """Verify user email with code"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.get("is_verified"):
        return VerificationResponse(
            message="Email already verified",
            is_verified=True
        )
    
    # Check if code matches
    if user.get("verification_code") != request_data.code:
        raise HTTPException(status_code=400, detail="Invalid verification code")
    
    # Check if code is expired
    code_expiry = user.get("verification_code_expiry")
    if not code_expiry or code_expiry < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Verification code expired. Please request a new code.")
    
    # Update user as verified
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {
            "$set": {"is_verified": True},
            "$unset": {"verification_code": "", "verification_code_expiry": ""}
        }
    )
    
    return VerificationResponse(
        message="Email verified successfully",
        is_verified=True
    )


@api_router.post("/auth/resend-verification", response_model=dict)
async def resend_verification_code(current_user_id: str = Depends(get_current_user)):
    """Resend verification code to user's email"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.get("is_verified"):
        raise HTTPException(status_code=400, detail="Email already verified")
    
    # Generate new code
    verification_code = generate_verification_code()
    code_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    # Update user with new code
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {
            "$set": {
                "verification_code": verification_code,
                "verification_code_expiry": code_expiry
            }
        }
    )
    
    # Send email
    email_sent = await send_verification_email(user["email"], verification_code, user["username"])
    if not email_sent:
        raise HTTPException(status_code=500, detail="Failed to send verification email")
    
    return {"message": "Verification code sent successfully"}


@api_router.get("/auth/verification-status", response_model=dict)
async def get_verification_status(current_user_id: str = Depends(get_current_user)):
    """Get user's verification status"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "is_verified": user.get("is_verified", False),
        "email": user["email"]
    }


@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, current_user_id: str = Depends(get_current_user)):
    """Logout user"""
    # Delete session from database
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.sessions.delete_many({"session_token": session_token})
    
    # Clear cookie
    response.delete_cookie(key="session_token", path="/")
    
    return {"message": "Logged out successfully"}


# ==================== PROFILE ROUTES ====================

@api_router.get("/profile/{user_id}", response_model=UserResponse)
async def get_profile(
    user_id: str, 
    request: Request,
    current_user_id: str = Depends(get_current_user_optional)
):
    """Get user profile by ID"""
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if they are friends
    response = user_to_response(user, current_user_id)
    
    if current_user_id and user_id != current_user_id:
        friendship = await db.friendships.find_one({
            "$or": [
                {"user_id": current_user_id, "friend_id": user_id},
                {"user_id": user_id, "friend_id": current_user_id}
            ]
        })
        
        if friendship:
            response.is_friend = friendship["status"] == "accepted"
            response.friend_request_status = friendship["status"]
        else:
            response.is_friend = False
            response.friend_request_status = "none"
    
    return response


@api_router.put("/profile", response_model=UserResponse)
async def update_profile(
    update: UserUpdate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Update current user's profile"""
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    
    if update_data:
        await db.users.update_one(
            {"_id": ObjectId(current_user_id)},
            {"$set": update_data}
        )
    
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    return user_to_response(user)


@api_router.put("/profile/username")
async def update_username(
    data: dict,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Update username"""
    new_username = data.get("username", "").strip()
    
    if not new_username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    # Check if username is already taken
    existing = await db.users.find_one({
        "username": new_username,
        "_id": {"$ne": ObjectId(current_user_id)}
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {"$set": {"username": new_username}}
    )
    
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    return user_to_response(user)


# ==================== RATING ROUTES ====================

@api_router.post("/ratings/{user_id}", response_model=RatingResponse)
async def rate_user(
    user_id: str,
    rating: RatingCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Rate another user (7-day cooldown between ratings)"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to rate users")
    
    if user_id == current_user_id:
        raise HTTPException(status_code=400, detail="Cannot rate yourself")
    
    # Check if user has already rated this person
    existing_rating = await db.ratings.find_one({
        "rated_user_id": user_id,
        "rater_user_id": current_user_id
    })
    
    if existing_rating:
        # Check if 7 days have passed since last rating
        last_rating_time = existing_rating.get("created_at")
        if last_rating_time:
            # Make sure last_rating_time is timezone-aware
            if last_rating_time.tzinfo is None:
                last_rating_time = last_rating_time.replace(tzinfo=timezone.utc)
            time_diff = datetime.now(timezone.utc) - last_rating_time
            days_passed = time_diff.total_seconds() / (24 * 3600)
            
            if days_passed < 7:
                days_remaining = 7 - days_passed
                raise HTTPException(
                    status_code=400, 
                    detail=f"You can only rate this user once every 7 days. Please wait {days_remaining:.1f} more days."
                )
        
        # Update existing rating (after cooldown period)
        await db.ratings.update_one(
            {"_id": existing_rating["_id"]},
            {"$set": {
                "stars": rating.stars,
                "comment": rating.comment,
                "created_at": datetime.now(timezone.utc)
            }}
        )
        result_id = existing_rating["_id"]
    else:
        # Create new rating
        rating_dict = {
            "rated_user_id": user_id,
            "rater_user_id": current_user_id,
            "stars": rating.stars,
            "comment": rating.comment,
            "created_at": datetime.now(timezone.utc)
        }
        result = await db.ratings.insert_one(rating_dict)
        result_id = result.inserted_id
    
    # Get rater info
    rater = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
    # Update user's average rating
    await update_user_rating(user_id)
    
    return RatingResponse(
        id=str(result_id),
        rated_user_id=user_id,
        rater_user_id=current_user_id,
        rater_username=rater["username"],
        rater_display_name=rater["display_name"],
        rater_profile_picture=rater.get("profile_picture"),
        stars=rating.stars,
        comment=rating.comment,
        created_at=datetime.now(timezone.utc)
    )


@api_router.get("/ratings/{user_id}/cooldown")
async def check_rating_cooldown(
    user_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Check if there's a cooldown for rating this user"""
    if user_id == current_user_id:
        return {
            "can_rate": False,
            "reason": "Cannot rate yourself",
            "days_remaining": None
        }
    
    # Check if user has already rated this person
    existing_rating = await db.ratings.find_one({
        "rated_user_id": user_id,
        "rater_user_id": current_user_id
    })
    
    if not existing_rating:
        return {
            "can_rate": True,
            "days_remaining": 0,
            "message": "You haven't rated this user yet"
        }
    
    # Check if 7 days have passed since last rating
    last_rating_time = existing_rating.get("created_at")
    if last_rating_time:
        # Make sure last_rating_time is timezone-aware
        if last_rating_time.tzinfo is None:
            last_rating_time = last_rating_time.replace(tzinfo=timezone.utc)
        time_diff = datetime.now(timezone.utc) - last_rating_time
        days_passed = time_diff.total_seconds() / (24 * 3600)
        
        if days_passed < 7:
            days_remaining = 7 - days_passed
            hours_remaining = days_remaining * 24
            return {
                "can_rate": False,
                "days_remaining": round(days_remaining, 1),
                "hours_remaining": round(hours_remaining, 1),
                "last_rated": last_rating_time.isoformat(),
                "message": f"You can rate this user again in {days_remaining:.1f} days"
            }
    
    return {
        "can_rate": True,
        "days_remaining": 0,
        "message": "You can rate this user now"
    }


@api_router.get("/ratings/{user_id}", response_model=List[RatingResponse])
async def get_ratings(user_id: str, request: Request):
    """Get all ratings for a user"""
    ratings = await db.ratings.find({"rated_user_id": user_id}).sort("created_at", -1).to_list(100)
    
    result = []
    for r in ratings:
        rater = await db.users.find_one({"_id": ObjectId(r["rater_user_id"])})
        result.append(RatingResponse(
            id=str(r["_id"]),
            rated_user_id=r["rated_user_id"],
            rater_user_id=r["rater_user_id"],
            rater_username=rater["username"],
            rater_display_name=rater["display_name"],
            rater_profile_picture=rater.get("profile_picture"),
            stars=r["stars"],
            comment=r.get("comment"),
            created_at=r["created_at"]
        ))
    
    return result


# ==================== FRIEND ROUTES ====================

@api_router.post("/friends/request")
async def send_friend_request(
    friend_req: FriendRequest,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Send friend request"""
    if friend_req.friend_id == current_user_id:
        raise HTTPException(status_code=400, detail="Cannot send friend request to yourself")
    
    # Check if friend exists
    friend = await db.users.find_one({"_id": ObjectId(friend_req.friend_id)})
    if not friend:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if request already exists
    existing = await db.friendships.find_one({
        "$or": [
            {"user_id": current_user_id, "friend_id": friend_req.friend_id},
            {"user_id": friend_req.friend_id, "friend_id": current_user_id}
        ]
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Friend request already exists")
    
    # Create friend request
    friendship_dict = {
        "user_id": current_user_id,
        "friend_id": friend_req.friend_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.friendships.insert_one(friendship_dict)
    
    return {"message": "Friend request sent"}


@api_router.put("/friends/accept/{request_id}")
async def accept_friend_request(
    request_id: str,
    req: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Accept friend request"""
    friendship = await db.friendships.find_one({"_id": ObjectId(request_id)})
    
    if not friendship:
        raise HTTPException(status_code=404, detail="Friend request not found")
    
    if friendship["friend_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.friendships.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"status": "accepted"}}
    )
    
    return {"message": "Friend request accepted"}


@api_router.put("/friends/reject/{request_id}")
async def reject_friend_request(
    request_id: str,
    req: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Reject friend request"""
    friendship = await db.friendships.find_one({"_id": ObjectId(request_id)})
    
    if not friendship:
        raise HTTPException(status_code=404, detail="Friend request not found")
    
    if friendship["friend_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.friendships.delete_one({"_id": ObjectId(request_id)})
    
    return {"message": "Friend request rejected"}


@api_router.get("/friends/list", response_model=List[FriendResponse])
async def get_friends(request: Request, current_user_id: str = Depends(get_current_user)):
    """Get list of accepted friends"""
    friendships = await db.friendships.find({
        "$or": [
            {"user_id": current_user_id, "status": "accepted"},
            {"friend_id": current_user_id, "status": "accepted"}
        ]
    }).to_list(None)
    
    result = []
    for f in friendships:
        friend_id = f["friend_id"] if f["user_id"] == current_user_id else f["user_id"]
        friend = await db.users.find_one({"_id": ObjectId(friend_id)})
        
        result.append(FriendResponse(
            id=str(f["_id"]),
            user_id=current_user_id,
            friend_id=friend_id,
            friend_username=friend["username"],
            friend_display_name=friend["display_name"],
            friend_profile_picture=friend.get("profile_picture"),
            status=f["status"],
            created_at=f["created_at"]
        ))
    
    return result


@api_router.get("/friends/pending", response_model=List[FriendResponse])
async def get_pending_requests(request: Request, current_user_id: str = Depends(get_current_user)):
    """Get pending friend requests (received)"""
    friendships = await db.friendships.find({
        "friend_id": current_user_id,
        "status": "pending"
    }).to_list(None)
    
    result = []
    for f in friendships:
        requester = await db.users.find_one({"_id": ObjectId(f["user_id"])})
        
        result.append(FriendResponse(
            id=str(f["_id"]),
            user_id=f["user_id"],
            friend_id=current_user_id,
            friend_username=requester["username"],
            friend_display_name=requester["display_name"],
            friend_profile_picture=requester.get("profile_picture"),
            status=f["status"],
            created_at=f["created_at"]
        ))
    
    return result


# ==================== SEARCH ROUTES ====================

@api_router.get("/search/users", response_model=List[UserResponse])
async def search_users(
    q: str,
    request: Request,
    current_user_id: str = Depends(get_current_user_optional)
):
    """Search users by name or username"""
    users = await db.users.find({
        "$or": [
            {"username": {"$regex": q, "$options": "i"}},
            {"display_name": {"$regex": q, "$options": "i"}}
        ]
    }).limit(20).to_list(20)
    
    return [user_to_response(u, current_user_id) for u in users]


# ==================== QR ROUTES ====================

@api_router.get("/qr/{user_id}")
async def get_qr_code(user_id: str):
    """Generate QR code for user profile"""
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create QR code with user profile URL
    qr_data = f"rateme://user/{user_id}"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {"qr_code": f"data:image/png;base64,{img_base64}"}


# ==================== GROUPS ROUTES ====================

@api_router.post("/groups/create", response_model=GroupResponse)
async def create_group(
    group: GroupCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Create a new group"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to create groups")
    
    group_dict = {
        "name": group.name,
        "description": group.description,
        "members": [current_user_id],
        "created_by": current_user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.groups.insert_one(group_dict)
    
    # Calculate average rating
    member_ratings = []
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if user:
        member_ratings.append(user.get("average_rating", 0.0))
    
    avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
    
    return GroupResponse(
        id=str(result.inserted_id),
        name=group_dict["name"],
        description=group_dict["description"],
        average_rating=round(avg_rating, 2),
        member_count=1,
        members=group_dict["members"],
        created_by=current_user_id,
        created_at=group_dict["created_at"]
    )


@api_router.post("/groups/join/{group_id}")
async def join_group(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Join a group"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to join groups")
    
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if already a member
    if current_user_id in group.get("members", []):
        raise HTTPException(status_code=400, detail="Already a member of this group")
    
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$push": {"members": current_user_id}}
    )
    
    return {"message": "Joined group successfully"}


@api_router.get("/groups", response_model=List[GroupResponse])
async def get_all_groups(request: Request):
    """Get all groups"""
    groups = await db.groups.find({}).to_list(100)
    
    result = []
    for g in groups:
        # Calculate average rating of members
        member_ratings = []
        for member_id in g.get("members", []):
            user = await db.users.find_one({"_id": ObjectId(member_id)})
            if user:
                member_ratings.append(user.get("average_rating", 0.0))
        
        avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
        
        result.append(GroupResponse(
            id=str(g["_id"]),
            name=g["name"],
            description=g.get("description"),
            average_rating=round(avg_rating, 2),
            member_count=len(g.get("members", [])),
            members=g.get("members", []),
            created_by=g["created_by"],
            created_at=g["created_at"]
        ))
    
    return result


@api_router.get("/groups/{group_id}", response_model=GroupResponse)
async def get_group(group_id: str, request: Request):
    """Get group details"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Calculate average rating
    member_ratings = []
    for member_id in group.get("members", []):
        user = await db.users.find_one({"_id": ObjectId(member_id)})
        if user:
            member_ratings.append(user.get("average_rating", 0.0))
    
    avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
    
    return GroupResponse(
        id=str(group["_id"]),
        name=group["name"],
        description=group.get("description"),
        average_rating=round(avg_rating, 2),
        member_count=len(group.get("members", [])),
        members=group.get("members", []),
        created_by=group["created_by"],
        created_at=group["created_at"]
    )


# ==================== COMPETITION ROUTES ====================

@api_router.post("/competitions/create", response_model=CompetitionResponse)
async def create_competition(
    comp: CompetitionCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Create a new 7-day competition"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to create competitions")
    
    start_date = datetime.now(timezone.utc)
    end_date = start_date + timedelta(days=7)
    
    comp_dict = {
        "name": comp.name,
        "start_date": start_date,
        "end_date": end_date,
        "status": "active",
        "participants": [{"user_id": current_user_id, "rating_at_start": 0.0}],
        "winner_id": None,
        "loser_id": None,
        "created_by": current_user_id
    }
    
    result = await db.competitions.insert_one(comp_dict)
    
    return CompetitionResponse(
        id=str(result.inserted_id),
        name=comp_dict["name"],
        start_date=comp_dict["start_date"],
        end_date=comp_dict["end_date"],
        status=comp_dict["status"],
        participants=comp_dict["participants"],
        created_by=current_user_id
    )


@api_router.post("/competitions/join/{comp_id}")
async def join_competition(
    comp_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Join an active competition"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to join competitions")
    
    comp = await db.competitions.find_one({"_id": ObjectId(comp_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if comp["status"] != "active":
        raise HTTPException(status_code=400, detail="Competition is not active")
    
    # Check if already joined
    if any(p["user_id"] == current_user_id for p in comp["participants"]):
        raise HTTPException(status_code=400, detail="Already joined this competition")
    
    # Get current rating
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
    await db.competitions.update_one(
        {"_id": ObjectId(comp_id)},
        {"$push": {"participants": {"user_id": current_user_id, "rating_at_start": user.get("average_rating", 0.0)}}}
    )
    
    return {"message": "Joined competition successfully"}


@api_router.get("/competitions/active", response_model=List[CompetitionResponse])
async def get_active_competitions(request: Request):
    """Get all active competitions"""
    comps = await db.competitions.find({
        "status": "active",
        "end_date": {"$gt": datetime.now(timezone.utc)}
    }).to_list(50)
    
    return [CompetitionResponse(
        id=str(c["_id"]),
        name=c["name"],
        start_date=c["start_date"],
        end_date=c["end_date"],
        status=c["status"],
        participants=c["participants"],
        winner_id=c.get("winner_id"),
        loser_id=c.get("loser_id"),
        created_by=c["created_by"]
    ) for c in comps]


@api_router.get("/competitions/my-competitions", response_model=List[CompetitionResponse])
async def get_my_competitions(request: Request, current_user_id: str = Depends(get_current_user)):
    """Get competitions user is participating in"""
    comps = await db.competitions.find({
        "participants.user_id": current_user_id
    }).to_list(50)
    
    return [CompetitionResponse(
        id=str(c["_id"]),
        name=c["name"],
        start_date=c["start_date"],
        end_date=c["end_date"],
        status=c["status"],
        participants=c["participants"],
        winner_id=c.get("winner_id"),
        loser_id=c.get("loser_id"),
        created_by=c["created_by"]
    ) for c in comps]


@api_router.get("/competitions/{comp_id}/participants", response_model=List[UserResponse])
async def get_competition_participants(comp_id: str, request: Request):
    """Get all participants in a competition with their current ratings"""
    comp = await db.competitions.find_one({"_id": ObjectId(comp_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    participants = []
    for p in comp["participants"]:
        user = await db.users.find_one({"_id": ObjectId(p["user_id"])})
        if user:
            participants.append(user_to_response(user))
    
    # Sort by average rating descending
    participants.sort(key=lambda x: x.average_rating, reverse=True)
    
    return participants


@api_router.post("/competitions/{comp_id}/invite")
async def invite_to_competition(
    comp_id: str,
    friend_req: FriendRequest,  # Reusing this model for user_id
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Invite a user to join a competition"""
    comp = await db.competitions.find_one({"_id": ObjectId(comp_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if comp["status"] != "active":
        raise HTTPException(status_code=400, detail="Competition is not active")
    
    # Check if invitee exists
    invitee = await db.users.find_one({"_id": ObjectId(friend_req.friend_id)})
    if not invitee:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already in competition
    if any(p["user_id"] == friend_req.friend_id for p in comp["participants"]):
        raise HTTPException(status_code=400, detail="User already in this competition")
    
    # Add user to competition
    await db.competitions.update_one(
        {"_id": ObjectId(comp_id)},
        {"$push": {"participants": {
            "user_id": friend_req.friend_id,
            "rating_at_start": invitee.get("average_rating", 0.0)
        }}}
    )
    
    return {"message": "User invited to competition successfully"}


@api_router.put("/competitions/{comp_id}")
async def update_competition(
    comp_id: str,
    comp_update: CompetitionCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Update competition name (only creator can edit)"""
    comp = await db.competitions.find_one({"_id": ObjectId(comp_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if comp["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only the creator can edit this competition")
    
    await db.competitions.update_one(
        {"_id": ObjectId(comp_id)},
        {"$set": {"name": comp_update.name}}
    )
    
    return {"message": "Competition updated successfully"}


@api_router.delete("/competitions/{comp_id}")
async def delete_competition(
    comp_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Delete competition (only creator can delete)"""
    comp = await db.competitions.find_one({"_id": ObjectId(comp_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if comp["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only the creator can delete this competition")
    
    await db.competitions.delete_one({"_id": ObjectId(comp_id)})
    
    return {"message": "Competition deleted successfully"}


@api_router.post("/competitions/finalize-ended")
async def finalize_ended_competitions(request: Request):
    """Check for ended competitions and award banners to winners/losers"""
    now = datetime.now(timezone.utc)
    
    # Find all active competitions that have ended
    ended_competitions = await db.competitions.find({
        "status": "active",
        "end_date": {"$lte": now}
    }).to_list(length=100)
    
    finalized_count = 0
    
    for comp in ended_competitions:
        # Get all participants with their current ratings
        participant_ratings = []
        for participant in comp["participants"]:
            user = await db.users.find_one({"_id": ObjectId(participant["user_id"])})
            if user:
                participant_ratings.append({
                    "user_id": participant["user_id"],
                    "rating": user.get("average_rating", 0.0)
                })
        
        if len(participant_ratings) < 2:
            # Mark competition as completed but don't award banners if less than 2 participants
            await db.competitions.update_one(
                {"_id": comp["_id"]},
                {"$set": {"status": "completed"}}
            )
            continue
        
        # Sort by rating
        participant_ratings.sort(key=lambda x: x["rating"], reverse=True)
        
        winner_id = participant_ratings[0]["user_id"]
        loser_id = participant_ratings[-1]["user_id"]
        
        # Set banners (expires in 7 days)
        banner_expiry = now + timedelta(days=7)
        
        # Award top-rated banner to winner
        await db.users.update_one(
            {"_id": ObjectId(winner_id)},
            {"$set": {"banner": "top-rated", "banner_expiry": banner_expiry}}
        )
        
        # Award try-harder banner to loser
        await db.users.update_one(
            {"_id": ObjectId(loser_id)},
            {"$set": {"banner": "try-harder", "banner_expiry": banner_expiry}}
        )
        
        # Mark competition as completed and store winner/loser
        await db.competitions.update_one(
            {"_id": comp["_id"]},
            {"$set": {
                "status": "completed",
                "winner_id": winner_id,
                "loser_id": loser_id
            }}
        )
        
        finalized_count += 1
    
    return {
        "message": f"Finalized {finalized_count} competitions",
        "finalized_count": finalized_count
    }


@api_router.post("/competitions/cleanup-banners")
async def cleanup_expired_banners(request: Request):
    """Remove expired banners from users"""
    now = datetime.now(timezone.utc)
    
    result = await db.users.update_many(
        {"banner_expiry": {"$lte": now}},
        {"$set": {"banner": None, "banner_expiry": None}}
    )
    
    return {
        "message": f"Cleaned up {result.modified_count} expired banners",
        "cleaned_count": result.modified_count
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
