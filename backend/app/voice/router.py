"""HTTP surface for the voice layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.auth import Principal, chat_principal
from app.timing import annotate as annotate_timings, turn as timed_turn
from app.voice.cache import cache_key, get_cache
from app.voice.client import VoiceUnavailable, get_client
from app.voice.config import ALLOWED_AUDIO_MIME, get_voice_settings
from app.voice.limiter import get_limiter
from app.voice.registry import Language, Persona, build_registry, resolve_profile
from app.voice.schemas import (
    PersonaVoice,
    SpeakRequest,
    TranscriptionResponse,
    VoiceConfigResponse,
    VoiceLimits,
)
from app.voice.speakable import has_many_numbers, speakable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

# What the client shows instead of a dead button when we cannot serve audio.
_FALLBACK = {"error": "voice_unavailable", "fallback": "browser"}

#: Distinct from `_FALLBACK` so an operator reading logs can tell an upstream
#: outage from a casting gap. The client's rule: an English-trained voice NEVER
#: speaks Spanish or French to a reader. No flag overrides this -- the override
#: is casting the voice, which is `VOICE_{PERSONA}_{ES|FR}` and a restart.
_UNCAST = {"error": "voice_uncast", "fallback": "browser"}


def _require_native(profile) -> None:
    """Native voice or no voice. Text is never affected.

    `resolve_profile` hands back whatever the registry could resolve, and for an
    ES/FR pair with no per-language id that is the persona's base voice: an
    English-trained model pushed through French text with the wrong accent and
    prosody. A French-speaking child hears it immediately, and it says, in the
    one channel built for the readers who cannot yet read well, that this
    product was not made for them. Silence degrades better: the player falls
    back exactly as it does when the upstream is down.
    """
    if not profile.native:
        logger.warning(
            "voice refused: %s/%s resolves only to an English-trained base "
            "voice. Cast VOICE_%s_%s to turn this pair's audio on.",
            profile.persona.value,
            profile.language.value,
            profile.persona.value.upper(),
            profile.language.value.upper(),
        )
        raise HTTPException(status_code=503, detail=_UNCAST)


@dataclass(frozen=True, slots=True)
class VoiceCaller:
    """Who is spending the voice budget, and the bucket it counts against.

    `key` is derived from a SIGNED token in both cases and is never anything the
    caller chose. That is the whole property this type exists to hold.
    """

    key: str
    #: None for a reader who has not signed in.
    user_id: str | None


async def require_voice_caller(
    authorization: str | None = Header(default=None),
    principal: Principal | None = Depends(chat_principal),
) -> VoiceCaller:
    """A caller whose spend can be attributed, or 401.

    All three of these endpoints were once open: no principal, no Authorization
    header read, and the only brake a window keyed on a thread id the caller
    supplied themselves. Locking them to an account fixed that and broke
    something else -- a signed-out visitor holds a `/v2/session` GRAPH token,
    which is a different type from an account token, so every guest got 401
    while the product told them otherwise. `/api/voice/config` advertises a
    `guest` voice in three languages and the composer says "hold Space to
    talk", against an API that refused them.

    So both tokens are accepted, and the budget is bucketed by whichever one
    arrived. A signed-out reader is bucketed by their SESSION id, taken from
    inside the signed token -- not from a body field, not from the socket
    address. Minting a session is itself rate limited (`session_mint_rate_limit`
    in `api/stream.py`), so the number of distinct buckets one visitor can open
    is bounded, which is what stops "a new session is a new budget" from being
    the old thread-id hole in new clothes.

    `chat_principal` rather than `optional_principal`: it allows the same grace
    on a just-expired token that chat does, so a reader mid-conversation is not
    cut off from the microphone for a few seconds of clock skew.
    """
    if principal is not None:
        return VoiceCaller(key=f"u:{principal.user_id}", user_id=str(principal.user_id))

    from app.auth import bearer_token
    from app.graph.identity import decode_session_token

    claims = decode_session_token(bearer_token(authorization))
    if claims is not None and claims.session_id:
        return VoiceCaller(key=f"s:{claims.session_id}", user_id=None)

    raise HTTPException(
        status_code=401, detail="A valid session is required to use voice."
    )


def _base_mime(raw: str | None) -> str:
    """The bare media type, without its RFC 2045 parameters."""
    return (raw or "").split(";", 1)[0].strip().lower()


@router.get("/config", response_model=VoiceConfigResponse)
def voice_config() -> VoiceConfigResponse:
    """Everything the client UI needs so it does not hardcode drifting numbers."""
    settings = get_voice_settings()
    registry = build_registry(settings)

    personas: list[PersonaVoice] = []
    for persona in Persona:
        languages = [lang for lang in Language if (persona, lang) in registry]
        if not languages:
            continue
        profile = registry[(persona, languages[0])]
        personas.append(
            PersonaVoice(
                persona=persona,
                languages=languages,
                speed=profile.settings.speed or 1.0,
                stability=profile.settings.stability or 0.5,
            )
        )

    return VoiceConfigResponse(
        enabled=settings.voice_enabled,
        personas=personas,
        languages=list(Language),
        limits=VoiceLimits(
            max_duration_seconds=settings.max_duration_seconds,
            max_file_size_bytes=settings.max_upload_bytes,
            allowed_mime_types=sorted(ALLOWED_AUDIO_MIME),
            max_transcriptions_per_window=settings.max_transcriptions_per_window,
            max_speech_per_window=settings.max_speech_per_window,
            rate_window_seconds=settings.rate_window_seconds,
        ),
        realtime_enabled=settings.voice_realtime_enabled,
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    voice_consent: bool = Form(...),
    language: str | None = Form(default=None),
    persona: str | None = Form(default=None),
    thread_id: str | None = Form(default=None),
    caller: VoiceCaller = Depends(require_voice_caller),
) -> TranscriptionResponse:
    settings = get_voice_settings()

    # Consent first: this audience starts at five, so no recording moves without it.
    if not voice_consent:
        raise HTTPException(
            status_code=403,
            detail="Voice consent is required before audio can be transcribed.",
        )

    if _base_mime(file.content_type) not in ALLOWED_AUDIO_MIME:
        raise HTTPException(
            status_code=415,
            # The unparsed value, so the log names exactly what the browser sent.
            detail=(
                f"Unsupported audio type {file.content_type!r}. "
                f"Supported: {', '.join(sorted(ALLOWED_AUDIO_MIME))}."
            ),
        )

    audio = await file.read()
    size = len(audio)
    if size == 0:
        raise HTTPException(status_code=400, detail="The audio file is empty.")
    if size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {settings.max_upload_bytes // 1_048_576} MB limit.",
        )
    # Byte proxy for the duration cap.
    if size > settings.duration_guard_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is longer than the {settings.max_duration_seconds:.0f} second limit.",
        )

    decision = get_limiter().check_transcription(caller.key)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many voice requests. Please wait a moment.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    hint = Language(language).value if language in {l.value for l in Language} else None

    try:
        transcript = await get_client().transcribe(
            audio, file.filename or "audio.webm", hint
        )
    except VoiceUnavailable:
        raise HTTPException(status_code=503, detail=_FALLBACK) from None
    finally:
        # Drop the reference promptly; nothing else ever held it.
        del audio

    # Deliberately no transcript text and no persona in the log.
    logger.info(
        "transcribe ok bytes=%d duration=%.1fs language=%s p=%.2f",
        size,
        transcript.duration_seconds,
        transcript.language_code,
        transcript.language_probability,
    )

    if (
        transcript.duration_seconds
        and transcript.duration_seconds > settings.max_duration_seconds
    ):
        raise HTTPException(
            status_code=413,
            detail=f"Audio is longer than the {settings.max_duration_seconds:.0f} second limit.",
        )

    # The transcript is a user message and nothing more.
    return TranscriptionResponse(
        text=transcript.text,
        language_code=transcript.language_code,
        language_probability=transcript.language_probability,
        duration_seconds=transcript.duration_seconds,
    )


@router.post("/speak")
async def speak(
    request: Request,
    body: SpeakRequest,
    caller: VoiceCaller = Depends(require_voice_caller),
) -> Response:
    """Text to audio."""
    with timed_turn(
        endpoint="/voice/speak",
        persona=body.persona.value,
        lang=body.language.value,
    ):
        return await _speak(request, body, caller)


async def _speak(request: Request, body: SpeakRequest, caller: VoiceCaller) -> Response:
    settings = get_voice_settings()

    if body.format.lower() != "mp3":
        raise HTTPException(
            status_code=400, detail=f"Unsupported format {body.format!r}. Use 'mp3'."
        )

    spoken = speakable(body.text, body.language, max_chars=settings.max_speakable_chars)
    if not spoken:
        raise HTTPException(status_code=400, detail="Nothing to say once the text was cleaned.")

    profile = resolve_profile(body.persona, body.language)
    _require_native(profile)
    # Figure-heavy lines go to the higher-quality model, which reads numbers better.
    model_id = (
        settings.tts_model_quality if has_many_numbers(body.text) else profile.model_id
    )

    key = cache_key(spoken, profile.voice_id, model_id, profile.settings)
    cache = get_cache()

    # Metered BEFORE the cache is consulted. A hit used to return audio without
    # touching the limiter, so a caller who kept asking for the same line was
    # never counted at all -- and the cache is shared, so the line only has to
    # be warm for somebody, not for them.
    decision = get_limiter().check_speech(caller.key)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many voice requests. Please wait a moment.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    # `aget`/`aput`, not `get`/`put`: a whole MP3 on the event loop would block it.
    if (cached := await cache.aget(key)) is not None:
        annotate_timings(cache_hit=True)
        return Response(
            content=cached,
            media_type="audio/mpeg",
            headers={"X-Voice-Cache": "hit", "Cache-Control": "private, max-age=86400"},
        )

    try:
        audio = await get_client().synthesise(
            spoken, profile.voice_id, model_id, profile.settings
        )
    except VoiceUnavailable:
        raise HTTPException(status_code=503, detail=_FALLBACK) from None

    await cache.aput(key, audio)
    logger.info(
        "speak ok persona=%s language=%s chars=%d model=%s bytes=%d",
        body.persona.value,
        body.language.value,
        len(spoken),
        model_id,
        len(audio),
    )

    return StreamingResponse(
        iter((audio,)),
        media_type="audio/mpeg",
        headers={"X-Voice-Cache": "miss", "Cache-Control": "private, max-age=86400"},
    )


@router.post("/speak-stream")
async def speak_stream(
    request: Request,
    body: SpeakRequest,
    caller: VoiceCaller = Depends(require_voice_caller),
) -> Response:
    """Text to audio, with the first byte sent before the last is synthesised."""
    with timed_turn(
        endpoint="/voice/speak-stream",
        persona=body.persona.value,
        lang=body.language.value,
    ):
        settings = get_voice_settings()

        if body.format.lower() != "mp3":
            raise HTTPException(
                status_code=400, detail=f"Unsupported format {body.format!r}. Use 'mp3'."
            )

        spoken = speakable(body.text, body.language, max_chars=settings.max_speakable_chars)
        if not spoken:
            raise HTTPException(
                status_code=400, detail="Nothing to say once the text was cleaned."
            )

        profile = resolve_profile(body.persona, body.language)
        _require_native(profile)
        model_id = (
            settings.tts_model_quality
            if has_many_numbers(body.text)
            else profile.model_id
        )

        headers = {
            "Cache-Control": "private, max-age=86400",
            # nginx buffers proxied responses by default, restoring the wait this removes.
            "X-Accel-Buffering": "no",
        }

        key = cache_key(spoken, profile.voice_id, model_id, profile.settings)
        cache = get_cache()

        # Metered before the cache is read, as in `_speak`. This one mattered
        # more: the hit returned above the limiter entirely, so replaying a warm
        # line was free and uncounted however many times it was asked for.
        decision = get_limiter().check_speech(caller.key)
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many voice requests. Please wait a moment.",
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )

        if (cached := await cache.aget(key)) is not None:
            annotate_timings(cache_hit=True)
            return Response(
                content=cached,
                media_type="audio/mpeg",
                headers={**headers, "X-Voice-Cache": "hit"},
            )

        # The first chunk is awaited before the response exists, so a failure can still 503.
        stream = get_client().synthesise_stream(
            spoken, profile.voice_id, model_id, profile.settings
        )
        try:
            first = await anext(stream)
        except VoiceUnavailable:
            raise HTTPException(status_code=503, detail=_FALLBACK) from None
        except StopAsyncIteration:
            raise HTTPException(status_code=503, detail=_FALLBACK) from None

        async def body_and_cache():
            # Teed: the reader hears chunks now, the cache gets the file only if it completes.
            collected = [first]
            clean = False
            try:
                yield first
                async for chunk in stream:
                    collected.append(chunk)
                    yield chunk
                clean = True
            finally:
                if clean:
                    audio = b"".join(collected)
                    await cache.aput(key, audio)
                    logger.info(
                        "speak-stream ok persona=%s language=%s chars=%d model=%s bytes=%d",
                        body.persona.value,
                        body.language.value,
                        len(spoken),
                        model_id,
                        len(audio),
                    )

        return StreamingResponse(
            body_and_cache(),
            media_type="audio/mpeg",
            headers={**headers, "X-Voice-Cache": "miss"},
        )


@router.post("/realtime-token")
async def realtime_token() -> Response:
    """Stretch goal, disabled by default."""
    if not get_voice_settings().voice_realtime_enabled:
        raise HTTPException(
            status_code=501,
            detail="Realtime voice is not enabled. Set VOICE_REALTIME_ENABLED=true.",
        )
    raise HTTPException(status_code=501, detail="Realtime token issuance is not implemented yet.")
