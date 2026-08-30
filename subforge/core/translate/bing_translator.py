"""Microsoft Azure AI Translator implementation."""

import os
import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Callable, List, Optional

import requests

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.base import BaseTranslator, logger
from subforge.core.translate.types import TargetLanguage, get_language_code
from subforge.core.utils.cache import generate_cache_key

DEFAULT_AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _normalize_translate_endpoint(endpoint: str) -> str:
    base = (endpoint or DEFAULT_AZURE_TRANSLATOR_ENDPOINT).strip().rstrip("/")
    if not base:
        base = DEFAULT_AZURE_TRANSLATOR_ENDPOINT
    if base.lower().endswith("/translate"):
        return base
    return f"{base}/translate"


class BingTranslator(BaseTranslator):
    """Official Microsoft Azure AI Translator client.

    The historical ``bing`` service identifier is retained so existing saved
    settings and subtitle jobs continue to resolve to this implementation.
    """

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        update_callback: Optional[Callable],
        use_cache: bool = True,
        api_key: str = "",
        region: str = "",
        endpoint: str = "",
        timeout: float = 30,
        max_retries: int = 5,
        session: requests.Session | None = None,
        cache_namespace: str = "",
    ):
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            update_callback=update_callback,
            use_cache=use_cache,
            cache_namespace=cache_namespace,
        )
        self.api_key = api_key.strip() or os.environ.get("AZURE_TRANSLATOR_KEY", "").strip()
        self.region = region.strip() or os.environ.get("AZURE_TRANSLATOR_REGION", "").strip()
        configured_endpoint = (
            endpoint.strip()
            or os.environ.get("AZURE_TRANSLATOR_ENDPOINT", "").strip()
            or DEFAULT_AZURE_TRANSLATOR_ENDPOINT
        )
        self.translate_endpoint = _normalize_translate_endpoint(configured_endpoint)
        self.timeout = max(1.0, float(timeout))
        self.max_retries = max(0, int(max_retries))
        self._owns_session = session is None
        self.session = session or requests.Session()
        self._session_closed = False

        if not self.api_key:
            self.stop()
            raise RuntimeError(
                "Microsoft Azure Translator API Key is not configured. "
                "Add it in Settings > Subtitle Processing > Translation Service."
            )

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        return headers

    def _ensure_thread_pool(self) -> None:
        """Restore owned HTTP resources when this translator is reused."""
        super()._ensure_thread_pool()
        if self._session_closed and self._owns_session:
            self.session = requests.Session()
            self._session_closed = False

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = str(response.headers.get("Retry-After", "")).strip()
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    now = parsedate_to_datetime(response.headers.get("Date", ""))
                    return min(60.0, max(0.0, (retry_at - now).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(30.0, 1.5 * (2**attempt))

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    message = error.get("message") or error.get("code")
                    if message:
                        return str(message)[:300]
        except (ValueError, TypeError):
            pass
        return str(getattr(response, "text", "") or "No response body")[:300]

    def _request_translations(self, texts: list[dict[str, str]], target_lang: str) -> list:
        params = {
            "api-version": "3.0",
            "to": target_lang,
            "includeSentenceLength": "true",
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if not self.is_running:
                raise RuntimeError("Microsoft Azure Translator request was cancelled")
            try:
                response = self.session.post(
                    self.translate_endpoint,
                    params=params,
                    headers=self._request_headers(),
                    json=texts,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30.0, 1.5 * (2**attempt)))
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "Microsoft Azure Translator returned invalid JSON"
                    ) from exc
                if not isinstance(payload, list):
                    raise RuntimeError(
                        "Microsoft Azure Translator returned an unexpected response"
                    )
                return payload

            detail = self._error_detail(response)
            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt >= self.max_retries:
                raise RuntimeError(
                    f"Microsoft Azure Translator HTTP {response.status_code}: {detail}"
                )
            delay = self._retry_delay(response, attempt)
            logger.warning(
                "Azure Translator request throttled or unavailable (HTTP %s); retrying in %.1fs",
                response.status_code,
                delay,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Microsoft Azure Translator network request failed after "
            f"{self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def test_connection(self) -> str:
        """Translate a minimal probe and return the translated text."""
        result = self._request_translations([{"Text": "Hello"}], "zh-Hans")
        return self._extract_translations(result, expected_count=1)[0]

    @staticmethod
    def _extract_translations(payload: list, expected_count: int) -> list[str]:
        if len(payload) != expected_count:
            raise RuntimeError(
                "Microsoft Azure Translator returned a different number of results "
                f"({len(payload)} instead of {expected_count})"
            )
        translated: list[str] = []
        for index, item in enumerate(payload, 1):
            try:
                text = item["translations"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Microsoft Azure Translator result {index} is missing translation text"
                ) from exc
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    f"Microsoft Azure Translator result {index} contains an empty translation"
                )
            translated.append(text.strip())
        return translated

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """Translate one subtitle batch through the official v3 endpoint."""
        if not subtitle_chunk:
            return subtitle_chunk
        target_lang = get_language_code(self.target_language, "bing")
        texts = [{"Text": data.original_text[:5000]} for data in subtitle_chunk]
        payload = self._request_translations(texts, target_lang)
        translated = self._extract_translations(payload, len(subtitle_chunk))
        for item, text in zip(subtitle_chunk, translated):
            item.translated_text = text
        return subtitle_chunk

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """Generate a cache key that cannot collide with the retired Edge API."""
        chunk_key = generate_cache_key(chunk)
        return f"AzureTranslatorV3:{chunk_key}:{self.target_language.value}"

    def stop(self):
        super().stop()
        if self._owns_session and not self._session_closed:
            try:
                self.session.close()
                self._session_closed = True
            except Exception:
                logger.warning("Failed to close Azure Translator session", exc_info=True)
