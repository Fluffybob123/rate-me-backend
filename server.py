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
from bson import ObjectId

from models import (
    UserCreate, UserLogin, UserResponse, UserUpdate, Token,
    RatingCreate, RatingResponse,
    FriendRequest, FriendResponse,
    CompetitionCreate, CompetitionResponse, CompetitionJoin
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

from fastapi import APIRouter
api_router = APIRouter(prefix="/api")


def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

@api_router.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    existing_user = await db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
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
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.users.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    del user_dict["_id"]
    del user_dict["password"]
    
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
    
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(db_user["_id"])})
    
    user_response = serialize_doc(db_user.copy())
    del user_response["password"]
    
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
    del user_response["password"]
    
    now = datetime.now(timezone.utc)
    banner = None
    if user.get("banner") and user.get("banner_expiry"):
        if user["banner_expiry"] > now:
            banner = user["banner"]
    
    user_response["banner"] = banner
    
    return UserResponse(**user_response)


async def update_user_rating(user_id: str):
    ratings = await db.ratings.find({"rated_user_id": user_id}).to_list(None)
    
    if ratings:
        total_stars = sum(r["stars"] for r in ratings)
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
    del user_response["password"]
    
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
            if friendship["status"] == "accepted":
                is_friend = True
            else:
                friend_request_status = friendship["status"]
    
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
    del user_response["password"]
    
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
        del user_response["password"]
        results.append(UserResponse(**user_response))
    
    return results

@api_router.post("/ratings/{user_id}", response_model=RatingResponse)
async def rate_user(
    user_id: str,
    rating: RatingCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
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
    
    rater = await db.users.find_one({"_id": ObjectId(current_user_id)})
    
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
        
        result.append(RatingResponse(
            id=str(rating["_id"]),
            rated_user_id=rating["rated_user_id"],
            rater_user_id=rating["rater_user_id"],
            rater_username=rater["username"],
            rater_display_name=rater["display_name"],
            rater_profile_picture=rater.get("profile_picture"),
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

@api_router.post("/competitions", response_model=CompetitionResponse)
async def create_competition(
    competition: CompetitionCreate,
    current_user_id: str = Depends(get_current_user)
):
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
    del competition_dict["_id"]
    
    return CompetitionResponse(**competition_dict)


@api_router.post("/competitions/join/{competition_id}")
async def join_competition(
    competition_id: str,
    current_user_id: str = Depends(get_current_user)
):
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
async def get_active_competitions(current_user_id: Optional[str] = Depends(get_current_user_optional)):
    competitions = await db.competitions.find({"status": "active"}).to_list(None)
    
    result = []
    for comp in competitions:
        comp_response = serialize_doc(comp)
        result.append(CompetitionResponse(**comp_response))
    
    return result


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
        p["average_rating"] = user.get("average_rating", 0.0)
        p["total_ratings"] = user.get("total_ratings", 0)
    
    sorted_participants = sorted(participants, key=lambda x: (-x["average_rating"], -x["total_ratings"]))
    
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
    for p in comp["participants"]:
        user = await db.users.find_one({"_id": ObjectId(p["user_id"])})
        if user:
            user_response = serialize_doc(user.copy())
            del user_response["password"]
            participants.append(UserResponse(**user_response))
    
    # Sort by average rating descending
    participants.sort(key=lambda x: x.average_rating, reverse=True)
    
    return participants


app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Rate Me API is running"}

