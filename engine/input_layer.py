import re
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageStat
from urllib.parse import urlparse

class InputProcessor:
    """
    Handles preprocessing of text, URLs, and images into a unified context.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans and normalizes text input."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _validate_url(url: str) -> str:
        """Returns an error message if the URL is invalid, else empty string."""
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"URL is missing a scheme — did you mean https://{url}?"
        if parsed.scheme not in ('http', 'https'):
            return f"Unsupported URL scheme '{parsed.scheme}'. Only http/https are supported."
        if not parsed.netloc:
            return "URL appears malformed — no domain found."
        return ""

    @staticmethod
    def fetch_url_content(url: str) -> tuple[str, str]:
        """
        Fetches and extracts core article text from a URL.
        Returns (content, warning). On failure, content is empty and warning describes the issue.
        """
        if not url or not url.strip():
            return "", ""

        warning = InputProcessor._validate_url(url.strip())
        if warning:
            return "", warning

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; BiasLens/1.0)'}
            response = requests.get(url.strip(), headers=headers, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'html' not in content_type and 'text' not in content_type:
                return "", f"URL returned non-readable content (type: {content_type.split(';')[0]})."

            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            text = InputProcessor.clean_text(soup.get_text(separator=' '))
            if not text:
                return "", "URL was fetched successfully but no readable text was found (may be a paywall or login page)."

            return text, ""

        except requests.exceptions.Timeout:
            return "", "URL request timed out after 10 seconds."
        except requests.exceptions.ConnectionError:
            return "", "Could not connect to the URL — check if the address is reachable."
        except requests.exceptions.HTTPError as e:
            return "", f"HTTP {e.response.status_code} {e.response.reason}."
        except Exception as e:
            return "", f"Unexpected error fetching URL: {str(e)}"

    @staticmethod
    def process_image(image: Image.Image) -> tuple[Image.Image | None, str]:
        """
        Preprocesses an image for the multimodal model.
        Returns (processed_image, warning). On failure, image is None and warning describes the issue.
        """
        if image is None:
            return None, ""

        try:
            if image.width < 32 or image.height < 32:
                return None, f"Image is too small ({image.width}×{image.height}px) to analyze meaningfully."

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Blank/uniform image detection — nearly zero variance means no visual content
            stat = ImageStat.Stat(image)
            if max(stat.stddev) < 5.0:
                return None, "Image appears blank or nearly uniform — no visual content to analyze."

            # Edge/Mobile: compress to avoid memory limits
            # [LITERT PLACEHOLDER]: In production, resize to exact quantized tensor shape (e.g., 224×224)
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            return image, ""

        except Exception as e:
            return None, f"Could not process image: {str(e)}"

    @classmethod
    def build_unified_context(cls, text: str, url: str, image: Image.Image) -> dict:
        """
        Takes raw inputs and returns a unified dictionary representing the reasoning context.
        Failures in URL fetch or image processing are captured as warnings, not exceptions,
        so partial input (e.g. text + broken URL) still proceeds to analysis.
        """
        cleaned_text = cls.clean_text(text)
        url_content, url_warning = cls.fetch_url_content(url)
        processed_image, image_warning = cls.process_image(image)

        warnings = [w for w in [url_warning, image_warning] if w]

        # --- CONTEXT FUSION LAYER ---
        # [LITERT PLACEHOLDER]: In mobile inference, this context is fused efficiently
        # into the quantized inputs without blowing up context lengths.
        combined_text_parts = []
        if cleaned_text:
            combined_text_parts.append(f"User Text:\n{cleaned_text}")
        if url_content:
            combined_text_parts.append(f"URL Content:\n{url_content}")

        combined_text = "\n\n".join(combined_text_parts)

        return {
            "has_image": processed_image is not None,
            "image": processed_image,
            "text": combined_text,
            "has_text": bool(combined_text.strip()),
            "warnings": warnings,
        }
