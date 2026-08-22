import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.deps import get_db, get_embedder, get_generator, get_reranker
from src.interfaces import (
    DatabaseProtocol,
    EmbedderProtocol,
    GeneratorProtocol,
    RerankerProtocol,
)
from src.routes.context import blob_by_filename, is_htmx, templates
from src.schemas import ChatRequest
from src.services.rag import run_rag
from src.storage.database import DBManager

router = APIRouter(tags=["chat"])


@router.get("/history")
async def get_history(db: DatabaseProtocol = Depends(get_db)):
    """Get chat history for the current user."""
    messages = db.get_messages()
    return {"messages": messages, "count": len(messages)}


@router.post("/chat", response_model=None)
async def chat(
    request: Request,
    db: DatabaseProtocol = Depends(get_db),
    embedder: EmbedderProtocol = Depends(get_embedder),
    generator: GeneratorProtocol = Depends(get_generator),
    reranker: RerankerProtocol = Depends(get_reranker),
):
    """Handle chat requests via JSON or HTMX form."""
    request_id = str(uuid.uuid4())

    # Parse request body based on content type
    is_json = False
    try:
        body = await request.json()
        req = ChatRequest(**body)
        is_json = True
    except Exception:
        # Fallback to form data (HTMX)
        form = await request.form()
        question = str(form.get("question", ""))
        top_k = 5
        filenames = None
        search_mode = "hybrid"
        persist = True
        rerank = True
    else:
        question = req.question
        top_k = req.top_k
        filenames = req.filenames
        search_mode = req.search_mode
        persist = req.persist
        rerank = req.rerank
    
    # Validate required fields
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required",
        )
    
    # Call RAG pipeline
    try:
        result = await run_rag(
            question=question,
            top_k=top_k,
            filenames=filenames,
            search_mode=search_mode,
            persist=persist,
            rerank=rerank,
            db=db,
            embedder=embedder,
            generator=generator,
            reranker=reranker,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}",
        )
    
    # Handle HTMX vs direct API response
    if is_htmx(request):
        documents = db.list_uploads()
        user_msg = {"role": "user", "content": question, "chunks": []}
        assistant_msg = {
            "role": "assistant",
            "content": result["answer"] or "",
            "chunks": result["chunks"],
            "latency": result["latency"],
        }
        return templates.TemplateResponse(
            request,
            "partials/chat_exchange.html",
            {
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "blob_by_filename": blob_by_filename(documents),
            },
            headers={"X-Request-ID": request_id},
        )

    return JSONResponse(
        content={
            "answer": result["answer"],
            "chunks": result["chunks"],
            "latency": result["latency"],
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )
