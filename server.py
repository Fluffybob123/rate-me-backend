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
    CompetitionCreate, CompetitionResponse, CompetitionJoin,
    GroupCreate, GroupResponse, GroupMemberInvite,
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
    # Add id explicitly for response and don't try to delete "_id" from the original dict
    user_dict["id"] = str(result.inserted_id)
    user_dict.pop("password", None)
    
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
    user_response.pop("password", None)
    
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
    rater_username = rater["username"] if rater else "Unknown"
    rater_display_name = rater["display_name"] if rater else "Unknown"
    rater_profile_picture = rater.get("profile_picture") if rater else None
    
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


    # ==================== GROUP ROUTES ====================

@api_router.post("/groups", response_model=GroupResponse)
async def create_group(
    group: GroupCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Create a new group"""
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
            members.append(UserResponse(
                id=str(user["_id"]),
                username=user["username"],
                email=user["email"],
                display_name=user["display_name"],
                bio=user.get("bio"),
                profile_picture=user.get("profile_picture"),
                average_rating=user.get("average_rating", 0.0),
                total_ratings=user.get("total_ratings", 0),
                created_at=user["created_at"],
                banner=user.get("banner"),
                banner_expiry=user.get("banner_expiry")
            ))
    
    return members


@api_router.post("/groups/{group_id}/join")
async def join_group(
    group_id: str,
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """Join a group"""
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if already a member
    if current_user_id in group.get("members", []):
        raise HTTPException(status_code=400, detail="Already a member of this group")
    
    # Add user to group
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$push": {"members": current_user_id}}
    )
    
    return {"message": "Successfully joined group"}


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
