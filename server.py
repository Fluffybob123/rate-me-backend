from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import os
from dotenv import load_dotenv
import qrcode
import io
import base64
import httpx
from bson import ObjectId
import resend
import random
import string

from models import (
    UserCreate, UserLogin, UserResponse, UserUpdate, Token,
    RatingCreate, RatingResponse,
    FriendRequest, FriendResponse,
    CompetitionCreate, CompetitionResponse, CompetitionJoin,
    GroupCreate, GroupResponse, GroupMemberInvite,
    NotificationPreferences, PushTokenRegister,
    VerifyEmailRequest, ResendCodeRequest, VerificationResponse,  # NEW
    SessionData, EmergentUserData
)
from auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, get_current_user_optional
)

load_dotenv()

app = FastAPI(title="Rate Me API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.rate_me

# Resend configuration
resend.api_key = "re_EDuB3JG7_293dQz9dvJo5WGvCEXFVKqrA"

from fastapi import APIRouter
api_router = APIRouter(prefix="/api")


def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    
    # Handle MongoDB datetime format
    if doc and "created_at" in doc:
        if isinstance(doc["created_at"], dict) and "$date" in doc["created_at"]:
            # Convert MongoDB $date format to datetime
            from dateutil import parser
            doc["created_at"] = parser.parse(doc["created_at"]["$date"])
        elif isinstance(doc["created_at"], str):
            # Convert string to datetime
            from dateutil import parser
            doc["created_at"] = parser.parse(doc["created_at"])
        # Ensure timezone-aware
        if doc["created_at"] and doc["created_at"].tzinfo is None:
            doc["created_at"] = doc["created_at"].replace(tzinfo=timezone.utc)
    
    # Handle banner_expiry the same way
    if doc and "banner_expiry" in doc and doc["banner_expiry"]:
        if isinstance(doc["banner_expiry"], dict) and "$date" in doc["banner_expiry"]:
            from dateutil import parser
            doc["banner_expiry"] = parser.parse(doc["banner_expiry"]["$date"])
        elif isinstance(doc["banner_expiry"], str):
            from dateutil import parser
            doc["banner_expiry"] = parser.parse(doc["banner_expiry"])
        # Ensure timezone-aware
        if doc["banner_expiry"].tzinfo is None:
            doc["banner_expiry"] = doc["banner_expiry"].replace(tzinfo=timezone.utc)
        
        # Check if banner is expired and clear it
        if doc["banner_expiry"] < datetime.now(timezone.utc):
            doc["banner"] = None
            doc["banner_expiry"] = None
    
    return doc


# ==================== EMAIL VERIFICATION HELPERS ====================

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
        
        response = resend.Emails.send(params)
        print(f"✅ Verification email sent successfully to {email}. Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Error sending verification email to {email}: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception details: {repr(e)}")
        return False


# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    existing_user = await db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Generate verification code
    verification_code = generate_verification_code()
    code_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    hashed_password = get_password_hash(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "display_name": user.display_name,
        "bio": "",
        "profile_picture": None,
        "average_rating": 0.0,
        "total_ratings": 0,
        "created_at": datetime.now(timezone.utc),
        "is_verified": False,
        "verification_code": verification_code,
        "verification_code_expiry": code_expiry
    }
    
    result = await db.users.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    user_dict.pop("password", None)
    
    # Send verification email
    email_sent = await send_verification_email(user.email, verification_code, user.username)
    if not email_sent:
        print(f"Failed to send verification email to {user.email}")
    
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user_dict)
    )


@api_router.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    # Try to find user by username first, then email
    db_user = await db.users.find_one({
        "$or": [
            {"username": user.username},
            {"email": user.username}  # Allow email in username field
        ]
    })
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if user has password (not OAuth user)
    if not db_user.get("password"):
        raise HTTPException(status_code=401, detail="This account uses Google login. Please sign in with Google.")
    
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(db_user["_id"])})
    
    user_response = serialize_doc(db_user.copy())
    user_response.pop("password", None)
    user_response["is_verified"] = db_user.get("is_verified", False)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user_response)
    )


@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user_id: str = Depends(get_current_user)):
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_response = serialize_doc(user.copy())
    user_response.pop("password", None)
    user_response["is_verified"] = user.get("is_verified", False)
    
    now = datetime.now(timezone.utc)
    banner = None
    if user.get("banner") and user.get("banner_expiry"):
        if user["banner_expiry"] > now:
            banner = user["banner"]
    
    user_response["banner"] = banner
    
    return UserResponse(**user_response)


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
    if code_expiry:
        # Handle timezone-naive datetime from MongoDB
        if code_expiry.tzinfo is None:
            code_expiry = code_expiry.replace(tzinfo=timezone.utc)
        
        if code_expiry < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Verification code expired. Please request a new code.")
    else:
        raise HTTPException(status_code=400, detail="No verification code found. Please request a new code.")
    
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


@api_router.delete("/auth/account")
async def delete_account(current_user_id: str = Depends(get_current_user)):
    """Delete user account and all associated data"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete all user's data
    try:
        # Delete ratings given by user
        await db.ratings.delete_many({"rater_user_id": current_user_id})
        
        # Delete ratings received by user
        await db.ratings.delete_many({"rated_user_id": current_user_id})
        
        # Remove user from all groups
        await db.groups.update_many(
            {"members": current_user_id},
            {"$pull": {"members": current_user_id}}
        )
        
        # Delete groups created by user
        await db.groups.delete_many({"created_by": current_user_id})
        
        # Delete group invitations
        await db.group_invitations.delete_many({
            "$or": [
                {"invited_by": current_user_id},
                {"invited_user_id": current_user_id}
            ]
        })
        
        # Remove user from competitions
        await db.competitions.update_many(
            {"participants.user_id": current_user_id},
            {"$pull": {"participants": {"user_id": current_user_id}}}
        )
        
        # Delete competitions created by user
        await db.competitions.delete_many({"created_by": current_user_id})
        
        # Delete competition invitations
        await db.competition_invitations.delete_many({
            "$or": [
                {"invited_by": current_user_id},
                {"invited_user_id": current_user_id}
            ]
        })
        
        # Delete notifications
        await db.notifications.delete_many({"user_id": current_user_id})
        
        # Finally, delete the user account
        await db.users.delete_one({"_id": ObjectId(current_user_id)})
        
        return {"message": "Account deleted successfully"}
    
    except Exception as e:
        print(f"Error deleting account {current_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete account")


# ==================== PROFILE ROUTES ====================

async def update_user_rating(user_id: str):
    ratings = await db.ratings.find({"rated_user_id": user_id}).to_list(None)
    
    if ratings:
        total_stars = sum(r.get("stars", 0) for r in ratings)
        avg_rating = total_stars / len(ratings)
    else:
        avg_rating = 0.0
    
    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(ratings)
        }}
    )

@api_router.get("/profile/{user_id}", response_model=UserResponse)
async def get_profile(
    user_id: str,
    request: Request,
    current_user_id: Optional[str] = Depends(get_current_user_optional)
):
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_response = serialize_doc(user.copy())
    user_response.pop("password", None)
    user_response["is_verified"] = user.get("is_verified", False)
    
    is_friend = False
    friend_request_status = "none"
    
    if current_user_id:
        friendship = await db.friends.find_one({
            "$or": [
                {"user_id": current_user_id, "friend_id": user_id},
                {"user_id": user_id, "friend_id": current_user_id}
            ]
        })
        
        if friendship:
            if friendship.get("status") == "accepted":
                is_friend = True
            else:
                friend_request_status = friendship.get("status", "none")
    
    user_response["is_friend"] = is_friend
    user_response["friend_request_status"] = friend_request_status
    
    now = datetime.now(timezone.utc)
    banner = None
    if user.get("banner") and user.get("banner_expiry"):
        if user["banner_expiry"] > now:
            banner = user["banner"]
    
    user_response["banner"] = banner
    
    return UserResponse(**user_response)


@api_router.put("/profile", response_model=UserResponse)
async def update_profile(
    user_update: UserUpdate,
    current_user_id: str = Depends(get_current_user)
):
    update_data = {k: v for k, v in user_update.dict().items() if v is not None}
    
    if update_data:
        await db.users.update_one(
            {"_id": ObjectId(current_user_id)},
            {"$set": update_data}
        )
    
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    user_response = serialize_doc(user.copy())
    user_response.pop("password", None)
    user_response["is_verified"] = user.get("is_verified", False)
    
    return UserResponse(**user_response)


@api_router.put("/profile/username")
async def update_username(
    new_username: str,
    current_user_id: str = Depends(get_current_user)
):
    existing = await db.users.find_one({"username": new_username})
    if existing and str(existing["_id"]) != current_user_id:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {"$set": {"username": new_username}}
    )
    
    return {"message": "Username updated successfully"}


@api_router.get("/search/users", response_model=List[UserResponse])
async def search_users(
    q: str,
    current_user_id: Optional[str] = Depends(get_current_user_optional)
):
    if len(q) < 2:
        return []
    
    users = await db.users.find({
        "$or": [
            {"username": {"$regex": q, "$options": "i"}},
            {"display_name": {"$regex": q, "$options": "i"}}
        ]
    }).limit(20).to_list(20)
    
    results = []
    for user in users:
        if str(user["_id"]) == current_user_id:
            continue
            
        user_response = serialize_doc(user.copy())
        user_response.pop("password", None)
        user_response["is_verified"] = user.get("is_verified", False)
        results.append(UserResponse(**user_response))
    
    return results


# ==================== RATING ROUTES (WITH VERIFICATION CHECK) ====================

@api_router.post("/ratings/{user_id}", response_model=RatingResponse)
async def rate_user(
    user_id: str,
    rating: RatingCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to rate users")
    
    if user_id == current_user_id:
        raise HTTPException(status_code=400, detail="Cannot rate yourself")
    
    existing_rating = await db.ratings.find_one({
        "rated_user_id": user_id,
        "rater_user_id": current_user_id
    })
    
    if existing_rating:
        last_rating_time = existing_rating.get("created_at")
        if last_rating_time:
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
        rating_dict = {
            "rated_user_id": user_id,
            "rater_user_id": current_user_id,
            "stars": rating.stars,
            "comment": rating.comment,
            "created_at": datetime.now(timezone.utc)
        }
        result = await db.ratings.insert_one(rating_dict)
        result_id = result.inserted_id
    
    await update_user_rating(user_id)
    
    # Get rater info
    rater = await db.users.find_one({"_id": ObjectId(current_user_id)})
    rater_username = rater["username"] if rater else "Unknown"
    rater_display_name = rater["display_name"] if rater else "Unknown"
    rater_profile_picture = rater.get("profile_picture") if rater else None
    
    # Send push notification for new review
    user_to_notify = await db.users.find_one({"_id": ObjectId(user_id)})
    if user_to_notify and user_to_notify.get("notification_preferences", {}).get("reviews", True):
        await send_push_notification(
            user_id,
            "New Review!",
            f"{rater_display_name} rated you {rating.stars} stars",
            {"type": "review", "rating_id": str(result_id)}
        )
    
    return RatingResponse(
        id=str(result_id),
        rated_user_id=user_id,
        rater_user_id=current_user_id,
        rater_username=rater_username,
        rater_display_name=rater_display_name,
        rater_profile_picture=rater_profile_picture,
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
    if user_id == current_user_id:
        return {
            "can_rate": False,
            "reason": "Cannot rate yourself",
            "days_remaining": None
        }
    
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
    
    last_rating_time = existing_rating.get("created_at")
    if last_rating_time:
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
async def get_ratings(user_id: str):
    ratings = await db.ratings.find({"rated_user_id": user_id}).sort("created_at", -1).to_list(None)
    
    result = []
    for rating in ratings:
        rater = await db.users.find_one({"_id": ObjectId(rating["rater_user_id"])})
        rater_username = rater["username"] if rater else "Unknown"
        rater_display_name = rater["display_name"] if rater else "Unknown"
        rater_profile_picture = rater.get("profile_picture") if rater else None
        
        result.append(RatingResponse(
            id=str(rating["_id"]),
            rated_user_id=rating["rated_user_id"],
            rater_user_id=rating["rater_user_id"],
            rater_username=rater_username,
            rater_display_name=rater_display_name,
            rater_profile_picture=rater_profile_picture,
            stars=rating["stars"],
            comment=rating.get("comment"),
            created_at=rating["created_at"]
        ))
    
    return result


@api_router.get("/qr/generate/{user_id}")
async def generate_qr_code(user_id: str):
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    qr_data = f"rateme://user/{user_id}"
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {"qr_code": f"data:image/png;base64,{img_base64}"}


# ==================== COMPETITION ROUTES (WITH VERIFICATION CHECK) ====================

@api_router.post("/competitions", response_model=CompetitionResponse)
async def create_competition(
    competition: CompetitionCreate,
    current_user_id: str = Depends(get_current_user)
):
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to create competitions")
    
    start_date = datetime.now(timezone.utc)
    end_date = start_date + timedelta(days=7)
    
    competition_dict = {
        "name": competition.name,
        "start_date": start_date,
        "end_date": end_date,
        "status": "active",
        "participants": [],
        "winner_id": None,
        "loser_id": None,
        "created_by": current_user_id
    }
    
    result = await db.competitions.insert_one(competition_dict)
    competition_dict["id"] = str(result.inserted_id)
    
    return CompetitionResponse(**competition_dict)


@api_router.post("/competitions/join/{competition_id}")
async def join_competition(
    competition_id: str,
    current_user_id: str = Depends(get_current_user)
):
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to join competitions")
    
    competition = await db.competitions.find_one({"_id": ObjectId(competition_id)})
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if competition["status"] != "active":
        raise HTTPException(status_code=400, detail="Competition is not active")
    
    already_joined = any(p["user_id"] == current_user_id for p in competition["participants"])
    if already_joined:
        raise HTTPException(status_code=400, detail="Already joined this competition")
    
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
    participant = {
        "user_id": current_user_id,
        "username": user["username"],
        "display_name": user["display_name"],
        "profile_picture": user.get("profile_picture"),
        "average_rating": user.get("average_rating", 0.0),
        "total_ratings": user.get("total_ratings", 0),
        "joined_at": datetime.now(timezone.utc)
    }
    
    await db.competitions.update_one(
        {"_id": ObjectId(competition_id)},
        {"$push": {"participants": participant}}
    )
    
    return {"message": "Successfully joined competition"}


@api_router.get("/competitions/active", response_model=List[CompetitionResponse])
async def get_active_competitions(request: Request):
    """Get all active competitions"""
    now = datetime.now(timezone.utc)
    
    # First, finalize any competitions that have ended
    ended_competitions = await db.competitions.find({
        "status": "active",
        "end_date": {"$lte": now}
    }).to_list(length=100)
    
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
        
        # Send push notifications to winner and loser
        try:
            winner = await db.users.find_one({"_id": ObjectId(winner_id)})
            loser = await db.users.find_one({"_id": ObjectId(loser_id)})
            
            if winner and winner.get("push_token") and winner.get("notifications_enabled", True):
                await send_push_notification(
                    winner_id,
                    "🏆 Competition Winner!",
                    f'You won the "{comp["name"]}" competition! You have the top-rated banner for 7 days.'
                )
            
            if loser and loser.get("push_token") and loser.get("notifications_enabled", True):
                await send_push_notification(
                    loser_id,
                    "Competition Ended",
                    f'The "{comp["name"]}" competition has ended. Keep improving your rating!'
                )
        except Exception as e:
            print(f"Failed to send competition result notifications: {e}")
        
        # Mark competition as completed and store winner/loser
        await db.competitions.update_one(
            {"_id": comp["_id"]},
            {"$set": {
                "status": "completed",
                "winner_id": winner_id,
                "loser_id": loser_id
            }}
        )
    
    # Now get only active competitions that haven't ended yet
    comps = await db.competitions.find({
        "status": "active",
        "end_date": {"$gt": now}
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
async def get_my_competitions(current_user_id: str = Depends(get_current_user)):
    competitions = await db.competitions.find({
        "participants.user_id": current_user_id
    }).to_list(None)
    
    result = []
    for comp in competitions:
        comp_response = serialize_doc(comp)
        result.append(CompetitionResponse(**comp_response))
    
    return result


@api_router.post("/competitions/{competition_id}/finalize")
async def finalize_competition(
    competition_id: str,
    current_user_id: str = Depends(get_current_user)
):
    competition = await db.competitions.find_one({"_id": ObjectId(competition_id)})
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if competition["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only creator can finalize")
    
    if competition["status"] != "active":
        raise HTTPException(status_code=400, detail="Competition already finalized")
    
    participants = competition["participants"]
    if len(participants) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 participants")
    
    for p in participants:
        user = await db.users.find_one({"_id": ObjectId(p["user_id"])})
        p["average_rating"] = user.get("average_rating", 0.0) if user else 0.0
        p["total_ratings"] = user.get("total_ratings", 0) if user else 0
    
    sorted_participants = sorted(participants, key=lambda x: (-x.get("average_rating", 0.0), -x.get("total_ratings", 0)))
    
    winner = sorted_participants[0]
    loser = sorted_participants[-1]
    
    banner_expiry = datetime.now(timezone.utc) + timedelta(days=7)
    
    await db.users.update_one(
        {"_id": ObjectId(winner["user_id"])},
        {"$set": {
            "banner": "top_rated",
            "banner_expiry": banner_expiry
        }}
    )
    
    await db.users.update_one(
        {"_id": ObjectId(loser["user_id"])},
        {"$set": {
            "banner": "try_harder",
            "banner_expiry": banner_expiry
        }}
    )
    
    await db.competitions.update_one(
        {"_id": ObjectId(competition_id)},
        {"$set": {
            "status": "completed",
            "winner_id": winner["user_id"],
            "loser_id": loser["user_id"],
            "participants": sorted_participants
        }}
    )
    
    # Send notifications to winner and loser
    winner_user = await db.users.find_one({"_id": ObjectId(winner["user_id"])})
    if winner_user and winner_user.get("notification_preferences", {}).get("competition_results", True):
        await send_push_notification(
            winner["user_id"],
            "🏆 You Won!",
            f"Congratulations! You won {competition['name']}",
            {"type": "competition_result", "competition_id": competition_id, "result": "winner"}
        )
    
    loser_user = await db.users.find_one({"_id": ObjectId(loser["user_id"])})
    if loser_user and loser_user.get("notification_preferences", {}).get("competition_results", True):
        await send_push_notification(
            loser["user_id"],
            "Competition Ended",
            f"{competition['name']} has ended. Keep improving!",
            {"type": "competition_result", "competition_id": competition_id, "result": "loser"}
        )
    
    return {"message": "Competition finalized", "winner": winner, "loser": loser}


@api_router.get("/competitions/{competition_id}", response_model=CompetitionResponse)
async def get_competition(competition_id: str):
    competition = await db.competitions.find_one({"_id": ObjectId(competition_id)})
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    comp_response = serialize_doc(competition)
    return CompetitionResponse(**comp_response)


@api_router.get("/competitions/{competition_id}/participants", response_model=List[UserResponse])
async def get_competition_participants(competition_id: str, request: Request):
    """Get all participants in a competition with their current ratings"""
    comp = await db.competitions.find_one({"_id": ObjectId(competition_id)})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    participants = []
    for p in comp.get("participants", []):
        user = await db.users.find_one({"_id": ObjectId(p["user_id"])})
        if user:
            user_response = serialize_doc(user.copy())
            user_response.pop("password", None)
            user_response["is_verified"] = user.get("is_verified", False)
            participants.append(UserResponse(**user_response))
    
    # Sort by average rating descending
    participants.sort(key=lambda x: x.average_rating, reverse=True)
    
    return participants


@api_router.post("/competitions/{comp_id}/invite")
async def invite_to_competition(
    comp_id: str,
    friend_req: FriendRequest,
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
    
    # Check if invitation already exists
    existing_invite = await db.competition_invitations.find_one({
        "competition_id": comp_id,
        "invited_user_id": friend_req.friend_id,
        "status": "pending"
    })
    
    if existing_invite:
        raise HTTPException(status_code=400, detail="Invitation already sent")
    
    # Create invitation
    invitation_doc = {
        "competition_id": comp_id,
        "competition_name": comp["name"],
        "invited_by": current_user_id,
        "invited_user_id": friend_req.friend_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.competition_invitations.insert_one(invitation_doc)
    
    # Send push notification for competition invitation
    invitee_user = await db.users.find_one({"_id": ObjectId(friend_req.friend_id)})
    if invitee_user and invitee_user.get("notification_preferences", {}).get("competition_invitations", True):
        await send_push_notification(
            friend_req.friend_id,
            "Competition Invitation",
            f"You've been invited to join {comp['name']}",
            {"type": "competition_invitation", "competition_id": comp_id}
        )
    
    return {"message": "Invitation sent successfully"}

@api_router.get("/competitions/invitations/pending")
async def get_pending_competition_invitations(
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get all pending competition invitations for current user"""
    invitations = await db.competition_invitations.find({
        "invited_user_id": current_user_id,
        "status": "pending"
    }).to_list(100)
    
    result = []
    for inv in invitations:
        # Get inviter info
        inviter = await db.users.find_one({"_id": ObjectId(inv["invited_by"])})
        inviter_name = inviter.get("display_name", "Unknown") if inviter else "Unknown"
        
        # Check if competition is still active
        comp = await db.competitions.find_one({"_id": ObjectId(inv["competition_id"])})
        if comp and comp["status"] == "active":
            result.append({
                "id": str(inv["_id"]),
                "competition_id": inv["competition_id"],
                "competition_name": inv["competition_name"],
                "invited_by": inv["invited_by"],
                "inviter_name": inviter_name,
                "created_at": inv["created_at"]
            })
    
    return result


@api_router.post("/competitions/invitations/{invitation_id}/accept")
async def accept_competition_invitation(
    invitation_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Accept a competition invitation"""
    invitation = await db.competition_invitations.find_one({"_id": ObjectId(invitation_id)})
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    
    if invitation["invited_user_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if invitation["status"] != "pending":
        raise HTTPException(status_code=400, detail="Invitation already processed")
    
    # Get competition
    comp = await db.competitions.find_one({"_id": ObjectId(invitation["competition_id"])})
    
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    if comp["status"] != "active":
        raise HTTPException(status_code=400, detail="Competition is no longer active")
    
    # Get user's current rating
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
    # Add user to competition
    await db.competitions.update_one(
        {"_id": ObjectId(invitation["competition_id"])},
        {"$push": {"participants": {
            "user_id": current_user_id,
            "username": user["username"],
            "display_name": user["display_name"],
            "profile_picture": user.get("profile_picture"),
            "average_rating": user.get("average_rating", 0.0),
            "total_ratings": user.get("total_ratings", 0),
            "joined_at": datetime.now(timezone.utc)
        }}}
    )
    
    # Update invitation status
    await db.competition_invitations.update_one(
        {"_id": ObjectId(invitation_id)},
        {"$set": {"status": "accepted"}}
    )
    
    return {"message": "Invitation accepted"}


@api_router.post("/competitions/invitations/{invitation_id}/reject")
async def reject_competition_invitation(
    invitation_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Reject a competition invitation"""
    invitation = await db.competition_invitations.find_one({"_id": ObjectId(invitation_id)})
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    
    if invitation["invited_user_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if invitation["status"] != "pending":
        raise HTTPException(status_code=400, detail="Invitation already processed")
    
    # Update invitation status
    await db.competition_invitations.update_one(
        {"_id": ObjectId(invitation_id)},
        {"$set": {"status": "rejected"}}
    )
    
    return {"message": "Invitation rejected"}

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
    
    # Delete all related invitations
    await db.competition_invitations.delete_many({"competition_id": comp_id})
    
    return {"message": "Competition deleted successfully"}


# ==================== NOTIFICATION ENDPOINTS ====================

@api_router.post("/notifications/register-token")
async def register_push_token(
    token_data: PushTokenRegister,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Register a push notification token for the user"""
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {"$set": {
            "push_token": token_data.push_token,
            "device_type": token_data.device_type,
            "token_registered_at": datetime.now(timezone.utc)
        }}
    )
    return {"message": "Push token registered successfully"}


@api_router.get("/notifications/preferences")
async def get_notification_preferences(
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get user's notification preferences"""
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Default preferences if not set
    prefs = user.get("notification_preferences", {
        "reviews": True,
        "group_invitations": True,
        "competition_invitations": True,
        "competition_results": True
    })
    
    return prefs


@api_router.put("/notifications/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferences,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Update user's notification preferences"""
    await db.users.update_one(
        {"_id": ObjectId(current_user_id)},
        {"$set": {
            "notification_preferences": preferences.dict()
        }}
    )
    
    return {"message": "Notification preferences updated successfully"}


async def send_push_notification(user_id: str, title: str, body: str, data: dict = None):
    """Helper function to send push notification to a user"""
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if not user or not user.get("push_token"):
        return
    
    # Store notification in database for history
    notification_doc = {
        "user_id": user_id,
        "title": title,
        "body": body,
        "data": data or {},
        "read": False,
        "created_at": datetime.now(timezone.utc)
    }
    await db.notifications.insert_one(notification_doc)
    
    # Send push notification via Expo Push API
    push_token = user.get("push_token")
    if push_token and push_token.startswith("ExponentPushToken"):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://exp.host/--/api/v2/push/send",
                    json={
                        "to": push_token,
                        "title": title,
                        "body": body,
                        "data": data or {},
                        "sound": "default",
                        "priority": "high",
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("data", {}).get("status") == "error":
                        print(f"Expo Push Error: {result.get('data', {}).get('message')}")
                else:
                    print(f"Failed to send push notification: {response.status_code}")
        except Exception as e:
            print(f"Error sending push notification: {str(e)}")


# ==================== GROUP ROUTES (WITH VERIFICATION CHECK) ====================

@api_router.post("/groups", response_model=GroupResponse)
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
    
    group_data = {
        "name": group.name,
        "description": group.description,
        "members": [current_user_id],
        "created_by": current_user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.groups.insert_one(group_data)
    
    # Calculate average rating
    user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    avg_rating = user.get("average_rating", 0.0) if user else 0.0
    
    return GroupResponse(
        id=str(result.inserted_id),
        name=group_data["name"],
        description=group_data.get("description"),
        average_rating=avg_rating,
        member_count=1,
        members=group_data["members"],
        created_by=group_data["created_by"],
        created_at=group_data["created_at"]
    )


@api_router.get("/groups", response_model=List[GroupResponse])
async def get_all_groups(request: Request, current_user_id: str = Depends(get_current_user)):
    """Get all groups"""
    groups = await db.groups.find().to_list(100)
    
    result = []
    for group in groups:
        # Calculate average rating from all members
        member_ratings = []
        for member_id in group.get("members", []):
            user = await db.users.find_one({"_id": ObjectId(member_id)})
            if user:
                member_ratings.append(user.get("average_rating", 0.0))
        
        avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
        
        result.append(GroupResponse(
            id=str(group["_id"]),
            name=group["name"],
            description=group.get("description"),
            average_rating=avg_rating,
            member_count=len(group.get("members", [])),
            members=group.get("members", []),
            created_by=group["created_by"],
            created_at=group["created_at"]
        ))
    
    return result


@api_router.get("/groups/search", response_model=List[GroupResponse])
async def search_groups(
    q: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Search groups by name"""
    groups = await db.groups.find({
        "name": {"$regex": q, "$options": "i"}
    }).to_list(50)
    
    result = []
    for group in groups:
        # Calculate average rating from all members
        member_ratings = []
        for member_id in group.get("members", []):
            user = await db.users.find_one({"_id": ObjectId(member_id)})
            if user:
                member_ratings.append(user.get("average_rating", 0.0))
        
        avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
        
        result.append(GroupResponse(
            id=str(group["_id"]),
            name=group["name"],
            description=group.get("description"),
            average_rating=avg_rating,
            member_count=len(group.get("members", [])),
            members=group.get("members", []),
            created_by=group["created_by"],
            created_at=group["created_at"]
        ))
    
    return result


@api_router.get("/groups/{group_id}", response_model=GroupResponse)
async def get_group(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get group details"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Calculate average rating from all members
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
        average_rating=avg_rating,
        member_count=len(group.get("members", [])),
        members=group.get("members", []),
        created_by=group["created_by"],
        created_at=group["created_at"]
    )


@api_router.get("/groups/{group_id}/members", response_model=List[UserResponse])
async def get_group_members(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get all members of a group"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    members = []
    for member_id in group.get("members", []):
        user = await db.users.find_one({"_id": ObjectId(member_id)})
        if user:
            user_response = serialize_doc(user.copy())
            user_response.pop("password", None)
            user_response["is_verified"] = user.get("is_verified", False)
            members.append(UserResponse(**user_response))
    
    return members


# ==================== GROUP JOIN REQUESTS ====================

@api_router.post("/groups/{group_id}/request-join")
async def request_join_group(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Request to join a group"""
    # Check if user is verified
    current_user = await db.users.find_one({"_id": ObjectId(current_user_id)})
    if not current_user or not current_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email to request to join groups")
    
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if already a member
    if current_user_id in group.get("members", []):
        raise HTTPException(status_code=400, detail="You are already a member of this group")
    
    # Check if request already exists
    existing_request = await db.group_join_requests.find_one({
        "group_id": group_id,
        "user_id": current_user_id,
        "status": "pending"
    })
    
    if existing_request:
        raise HTTPException(status_code=400, detail="You have already requested to join this group")
    
    # Create join request
    join_request = {
        "group_id": group_id,
        "group_name": group["name"],
        "user_id": current_user_id,
        "username": current_user["username"],
        "display_name": current_user["display_name"],
        "profile_picture": current_user.get("profile_picture"),
        "status": "pending",
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.group_join_requests.insert_one(join_request)
    
    # Send notification to group creator
    creator = await db.users.find_one({"_id": ObjectId(group["created_by"])})
    if creator and creator.get("notification_preferences", {}).get("group_invitations", True):
        await send_push_notification(
            group["created_by"],
            "Group Join Request",
            f"{current_user['display_name']} wants to join {group['name']}",
            {"type": "group_join_request", "group_id": group_id}
        )
    
    return {"message": "Join request sent successfully"}


@api_router.get("/groups/{group_id}/join-requests")
async def get_group_join_requests(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get all pending join requests for a group (creator only)"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Only group creator can view join requests
    if group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can view join requests")
    
    requests = await db.group_join_requests.find({
        "group_id": group_id,
        "status": "pending"
    }).to_list(100)
    
    result = []
    for req in requests:
        result.append({
            "id": str(req["_id"]),
            "group_id": req["group_id"],
            "user_id": req["user_id"],
            "username": req["username"],
            "display_name": req["display_name"],
            "profile_picture": req.get("profile_picture"),
            "created_at": req["created_at"]
        })
    
    return result


@api_router.get("/groups/join-requests/all")
async def get_all_join_requests_for_user(
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get all pending join requests for all groups created by current user"""
    # Find all groups created by user
    groups = await db.groups.find({"created_by": current_user_id}).to_list(100)
    group_ids = [str(g["_id"]) for g in groups]
    
    if not group_ids:
        return []
    
    # Find all pending join requests for those groups
    requests = await db.group_join_requests.find({
        "group_id": {"$in": group_ids},
        "status": "pending"
    }).to_list(100)
    
    result = []
    for req in requests:
        result.append({
            "id": str(req["_id"]),
            "group_id": req["group_id"],
            "group_name": req["group_name"],
            "user_id": req["user_id"],
            "username": req["username"],
            "display_name": req["display_name"],
            "profile_picture": req.get("profile_picture"),
            "created_at": req["created_at"]
        })
    
    return result


@api_router.post("/groups/join-requests/{request_id}/accept")
async def accept_join_request(
    request_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Accept a join request (creator only)"""
    join_request = await db.group_join_requests.find_one({"_id": ObjectId(request_id)})
    
    if not join_request:
        raise HTTPException(status_code=404, detail="Join request not found")
    
    # Get group and verify creator
    group = await db.groups.find_one({"_id": ObjectId(join_request["group_id"])})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can accept join requests")
    
    if join_request["status"] != "pending":
        raise HTTPException(status_code=400, detail="Request already processed")
    
    # Add user to group
    await db.groups.update_one(
        {"_id": ObjectId(join_request["group_id"])},
        {"$push": {"members": join_request["user_id"]}}
    )
    
    # Update request status
    await db.group_join_requests.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"status": "accepted"}}
    )
    
    # Send notification to user
    user = await db.users.find_one({"_id": ObjectId(join_request["user_id"])})
    if user and user.get("notification_preferences", {}).get("group_invitations", True):
        await send_push_notification(
            join_request["user_id"],
            "Join Request Accepted",
            f"Your request to join {group['name']} was accepted!",
            {"type": "group_join_accepted", "group_id": join_request["group_id"]}
        )
    
    return {"message": "Join request accepted"}


@api_router.post("/groups/join-requests/{request_id}/reject")
async def reject_join_request(
    request_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Reject a join request (creator only)"""
    join_request = await db.group_join_requests.find_one({"_id": ObjectId(request_id)})
    
    if not join_request:
        raise HTTPException(status_code=404, detail="Join request not found")
    
    # Get group and verify creator
    group = await db.groups.find_one({"_id": ObjectId(join_request["group_id"])})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can reject join requests")
    
    if join_request["status"] != "pending":
        raise HTTPException(status_code=400, detail="Request already processed")
    
    # Update request status
    await db.group_join_requests.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"status": "rejected"}}
    )
    
    return {"message": "Join request rejected"}


# ==================== GROUPS LIST ROUTES ====================


@api_router.post("/groups/{group_id}/invite")
async def invite_to_group(
    group_id: str,
    invite: GroupMemberInvite,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Invite a user to a group (only creator can invite)"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if current user is the creator
    if group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can invite members")
    
    # Check if user exists
    invited_user = await db.users.find_one({"_id": ObjectId(invite.user_id)})
    if not invited_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user is already a member
    if invite.user_id in group.get("members", []):
        raise HTTPException(status_code=400, detail="User is already a member")
    
    # Check if invitation already exists
    existing_invite = await db.group_invitations.find_one({
        "group_id": group_id,
        "invited_user_id": invite.user_id,
        "status": "pending"
    })
    
    if existing_invite:
        raise HTTPException(status_code=400, detail="Invitation already sent")
    
    # Create invitation (don't add to members yet)
    invitation_doc = {
        "group_id": group_id,
        "group_name": group["name"],
        "invited_by": current_user_id,
        "invited_user_id": invite.user_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.group_invitations.insert_one(invitation_doc)
    
    # Send push notification for group invitation
    invitee_user = await db.users.find_one({"_id": ObjectId(invite.user_id)})
    if invitee_user and invitee_user.get("notification_preferences", {}).get("group_invitations", True):
        await send_push_notification(
            invite.user_id,
            "Group Invitation",
            f"You've been invited to join {group['name']}",
            {"type": "group_invitation", "group_id": group_id}
        )
    
    return {"message": "Invitation sent successfully"}

@api_router.get("/groups/invitations/pending")
async def get_pending_invitations(
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Get all pending group invitations for current user"""
    invitations = await db.group_invitations.find({
        "invited_user_id": current_user_id,
        "status": "pending"
    }).to_list(100)
    
    result = []
    for inv in invitations:
        # Get inviter info
        inviter = await db.users.find_one({"_id": ObjectId(inv["invited_by"])})
        inviter_name = inviter.get("display_name", "Unknown") if inviter else "Unknown"
        
        result.append({
            "id": str(inv["_id"]),
            "group_id": inv["group_id"],
            "group_name": inv["group_name"],
            "invited_by": inv["invited_by"],
            "inviter_name": inviter_name,
            "created_at": inv["created_at"]
        })
    
    return result


@api_router.post("/groups/invitations/{invitation_id}/accept")
async def accept_group_invitation(
    invitation_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Accept a group invitation"""
    invitation = await db.group_invitations.find_one({"_id": ObjectId(invitation_id)})
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    
    if invitation["invited_user_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if invitation["status"] != "pending":
        raise HTTPException(status_code=400, detail="Invitation already processed")
    
    # Add user to group
    await db.groups.update_one(
        {"_id": ObjectId(invitation["group_id"])},
        {"$push": {"members": current_user_id}}
    )
    
    # Update invitation status
    await db.group_invitations.update_one(
        {"_id": ObjectId(invitation_id)},
        {"$set": {"status": "accepted"}}
    )
    
    return {"message": "Invitation accepted"}


@api_router.post("/groups/invitations/{invitation_id}/reject")
async def reject_group_invitation(
    invitation_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Reject a group invitation"""
    invitation = await db.group_invitations.find_one({"_id": ObjectId(invitation_id)})
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    
    if invitation["invited_user_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if invitation["status"] != "pending":
        raise HTTPException(status_code=400, detail="Invitation already processed")
    
    # Update invitation status
    await db.group_invitations.update_one(
        {"_id": ObjectId(invitation_id)},
        {"$set": {"status": "rejected"}}
    )
    
    return {"message": "Invitation rejected"}


@api_router.put("/groups/{group_id}", response_model=GroupResponse)
async def update_group(
    group_id: str,
    group: GroupCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Update a group (only creator can update)"""
    existing_group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not existing_group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if current user is the creator
    if existing_group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can update the group")
    
    # Update group
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$set": {
            "name": group.name,
            "description": group.description
        }}
    )
    
    # Recompute average rating from all members
    member_ratings = []
    for member_id in existing_group.get("members", []):
        user = await db.users.find_one({"_id": ObjectId(member_id)})
        if user:
            member_ratings.append(user.get("average_rating", 0.0))
    
    avg_rating = sum(member_ratings) / len(member_ratings) if member_ratings else 0.0
    
    return GroupResponse(
        id=str(existing_group["_id"]),
        name=group.name,
        description=group.description,
        average_rating=avg_rating,
        member_count=len(existing_group.get("members", [])),
        members=existing_group.get("members", []),
        created_by=existing_group["created_by"],
        created_at=existing_group["created_at"]
    )
    

@api_router.delete("/groups/{group_id}")
async def delete_group(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Delete a group (only creator can delete)"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if current user is the creator
    if group["created_by"] != current_user_id:
        raise HTTPException(status_code=403, detail="Only group creator can delete the group")
    
    await db.groups.delete_one({"_id": ObjectId(group_id)})
    
    return {"message": "Group deleted successfully"}


app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Rate Me API is running"}
