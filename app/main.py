import os
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    Configuration,
    ReplyMessageRequest,
    TextMessage,
)

from app.response_message import generate_reply_message
from app.retrieval_utils import ask_question_to_rag, store_user_response, store_user_question, check_and_update_question_limit
from app.quick_reply import handle_quick_reply
from app.content_filter import check_content_safety

app = FastAPI()

load_dotenv(override=True)

def get_secret_value(secret_name, default=None):
    secret_path = f"/secrets/{secret_name}"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()
    return os.getenv(secret_name, default)

get_access_token = get_secret_value('LINE_CHANNEL_ACCESS_TOKEN')
get_channel_secret = get_secret_value('LINE_CHANNEL_SECRET')
print(f"LINE_CHANNEL_SECRET: {get_channel_secret}")

configuration = Configuration(access_token=get_access_token)
handler = WebhookHandler(channel_secret=get_channel_secret)

@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    body_str = body.decode('utf-8')
    print(f"Received body: {body_str}")

    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def on_message_event(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # ✅ ลองให้ quick reply ทำงานก่อน
        quick_reply = handle_quick_reply(event)
        if quick_reply:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[quick_reply]
                )
            )
            return

        # ถ้าไม่ใช่ quick reply, ไปถาม RAG ปกติ
        reply_message = generate_reply_message(event)
        if not reply_message:
            return None

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[reply_message]
            )
        )


# ------------------------
# ✅ RAG Endpoint /ask
# ------------------------
class AskRequest(BaseModel):
    user_id: str
    question: str

@app.post("/ask")
async def ask_route(req: AskRequest):
    # 🛡️ ตรวจสอบความปลอดภัยของเนื้อหาก่อน
    is_safe, safety_message = check_content_safety(req.question)
    if not is_safe:
        print(f"🚫 Content filtered for user {req.user_id}: {safety_message}")
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=req.question,
            user_id=req.user_id,
            context_data={"endpoint": "/ask", "filtered": True}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=req.question,
            answer=safety_message,
            user_id=req.user_id,
            response_type="content_filtered",
            context_data={"filter_reason": "unsafe_content", "endpoint": "/ask"}
        )
        
        return {"answer": safety_message}
    
    # ตรวจสอบจำนวนคำถามต่อเนื่อง
    is_allowed, current_count, limit_message = check_and_update_question_limit(req.user_id, max_questions=3)
    if not is_allowed:
        print(f"🚫 Question limit exceeded for user {req.user_id}: {current_count}/3")
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=req.question,
            user_id=req.user_id,
            context_data={"endpoint": "/ask", "limit_exceeded": True}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=req.question,
            answer=limit_message,
            user_id=req.user_id,
            response_type="question_limit_exceeded",
            context_data={"question_count": current_count, "max_questions": 3, "endpoint": "/ask"}
        )
        
        return {"answer": limit_message}
    
    # บันทึกคำถามใน user_profiles
    store_user_question(
        question=req.question,
        user_id=req.user_id,
        context_data={"endpoint": "/ask"}
    )
    
    answer = ask_question_to_rag(req.question, req.user_id)
    
    # บันทึกคำตอบใน collection astrobot (ask_question_to_rag จะบันทึกเองแล้ว แต่เพิ่มข้อมูล endpoint)
    store_user_response(
        question=req.question,
        answer=answer,
        user_id=req.user_id,
        response_type="api_response",
        context_data={"endpoint": "/ask"}
    )
    
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
