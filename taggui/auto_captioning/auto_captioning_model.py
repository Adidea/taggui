import base64
import json
import mimetypes
import re
import os
from pathlib import Path
import time
from datetime import datetime
from io import BytesIO
from urllib import error, request

from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose

import auto_captioning.captioning_thread as captioning_thread
from utils.image import Image


def replace_template_variable(match: re.Match, image: Image) -> str:
    template_variable = match.group(0)[1:-1].lower()
    if template_variable == 'tags':
        return ', '.join(image.tags)
    if template_variable == 'name':
        return image.path.stem
    if template_variable in ('directory', 'folder'):
        return image.path.parent.name
    return ''


def replace_template_variables(text: str, image: Image) -> str:
    # Replace template variables inside curly braces that are not escaped.
    text = re.sub(r'(?<!\\){[^{}]+(?<!\\)}',
                  lambda match: replace_template_variable(match, image), text)
    # Unescape escaped curly braces.
    text = re.sub(r'\\([{}])', r'\1', text)
    return text


def split_escaped_text(text: str, separator: str) -> list[str]:
    if not text.strip():
        return []
    values = re.split(rf'(?<!\\){separator}', text)
    return [value.strip() for value in values if value.strip()]


class AutoCaptioningModel:
    image_mode = 'RGB'

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        self.thread = captioning_thread_
        self.caption_settings = caption_settings
        self.model_id = caption_settings['model_id']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        self.remove_tag_separators = caption_settings['remove_tag_separators']
        self.bad_words_string = caption_settings['bad_words']
        self.forced_words_string = caption_settings['forced_words']
        self.generation_parameters = caption_settings['generation_parameters']
        self.api_base_url = caption_settings['api_base_url'].strip()
        self.api_key = caption_settings['api_key'].strip()
        self.request_timeout_seconds = caption_settings['request_timeout']

    def get_error_message(self) -> str | None:
        if not self.api_base_url:
            return ('Set an OpenAI-compatible API base URL in Settings before '
                    'starting auto-captioning.')
        if not self.model_id.strip():
            return 'The `Model` field cannot be empty.'
        if self.request_timeout_seconds <= 0:
            return 'The request timeout must be greater than 0 seconds.'
        return None

    @staticmethod
    def get_captioning_start_datetime_string(
            captioning_start_datetime: datetime) -> str:
        return captioning_start_datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return ('Captioning with remote API... '
                    f'(start time: {captioning_start_datetime_string})')
        return 'Captioning with remote API...'

    @staticmethod
    def get_default_prompt() -> str:
        return ''

    @staticmethod
    def format_prompt(prompt: str) -> str:
        return prompt

    def get_image_prompt(self, image: Image) -> str:
        if self.prompt:
            image_prompt = self.replace_template_variables(self.prompt, image)
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        image_prompt = self.format_prompt(image_prompt)
        return image_prompt

    def load_image(self, image: Image) -> PilImage:
        pil_image = PilImage.open(image.path)
        # Rotate the image according to the orientation tag.
        pil_image = exif_transpose(pil_image)
        pil_image = pil_image.convert(self.image_mode)
        return pil_image

    def get_chat_completion_url(self) -> str:
        url = self.api_base_url.rstrip('/')
        if url.endswith('/chat/completions'):
            return url
        return f'{url}/chat/completions'

    def get_image_data_url(self, image: Image) -> str:
        pil_image = self.load_image(image)
        original_mime_type, _ = mimetypes.guess_type(image.path.name)
        output_mime_type = original_mime_type or 'image/png'
        image_format = 'PNG'
        if output_mime_type in ('image/jpeg', 'image/jpg'):
            image_format = 'JPEG'
        elif output_mime_type == 'image/webp':
            image_format = 'WEBP'
        else:
            output_mime_type = 'image/png'
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format=image_format)
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode(
            'ascii')
        return f'data:{output_mime_type};base64,{encoded_image}'

    def get_bad_words(self) -> list[str]:
        words = split_escaped_text(self.bad_words_string, ',')
        return [word.replace(r'\,', ',') for word in words]

    def get_forced_word_groups(self) -> list[list[str]]:
        word_groups = split_escaped_text(self.forced_words_string, ',')
        groups = []
        for word_group in word_groups:
            words = split_escaped_text(word_group.replace(r'\,', ','), r'\|')
            words = [word.replace(r'\|', '|') for word in words]
            if words:
                groups.append(words)
        return groups

    def get_prompt_text(self, image_prompt: str) -> str:
        prompt_parts = []
        if image_prompt:
            prompt_parts.append(image_prompt)
        if self.caption_start.strip():
            prompt_parts.append(
                f'Start the caption with: {self.caption_start.strip()}')
        bad_words = self.get_bad_words()
        if bad_words:
            prompt_parts.append(
                f'Avoid these words or phrases: {", ".join(bad_words)}')
        forced_word_groups = self.get_forced_word_groups()
        if forced_word_groups:
            required_phrases = []
            for group in forced_word_groups:
                if len(group) == 1:
                    required_phrases.append(group[0])
                else:
                    required_phrases.append(f'one of: {", ".join(group)}')
            prompt_parts.append(
                f'Include these words or phrases: {"; ".join(required_phrases)}')
        return '\n\n'.join(prompt_parts)

    def get_request_payload(self, image_prompt: str, image: Image) -> dict:
        generation_parameters = self.generation_parameters
        if generation_parameters['do_sample']:
            temperature = generation_parameters['temperature']
        else:
            temperature = 0
        prompt_text = self.get_prompt_text(image_prompt)
        payload = {
            'model': self.model_id,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt_text},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': self.get_image_data_url(image)
                            }
                        }
                    ]
                }
            ],
            'max_tokens': generation_parameters['max_new_tokens'],
            'temperature': temperature,
            'top_p': generation_parameters['top_p'],
            'stream': False,
            # Many OpenAI-compatible backends accept these extras even though
            # they are not part of the core API schema.
            'min_tokens': generation_parameters['min_new_tokens'],
            'top_k': generation_parameters['top_k'],
            'repetition_penalty': generation_parameters['repetition_penalty'],
            'repeat_penalty': generation_parameters['repetition_penalty'],
            'no_repeat_ngram_size':
                generation_parameters['no_repeat_ngram_size'],
            'num_beams': generation_parameters['num_beams'],
            'length_penalty': generation_parameters['length_penalty'],
            'do_sample': generation_parameters['do_sample']
        }
        return payload

    def make_request(self, payload: dict) -> dict:
        request_headers = {'Content-Type': 'application/json'}
        if self.api_key:
            request_headers['Authorization'] = f'Bearer {self.api_key}'
        request_data = json.dumps(payload).encode('utf-8')
        http_request = request.Request(
            url=self.get_chat_completion_url(), data=request_data,
            headers=request_headers, method='POST')
        
        # Retry logic with exponential backoff for transient errors
        max_retries = 5
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                with request.urlopen(
                        http_request, timeout=self.request_timeout_seconds) as response:
                    response_body = response.read().decode('utf-8')
                    
                # Add small delay after successful request to avoid overwhelming server
                time.sleep(0.5)
                
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError as exception:
                    raise RuntimeError(
                        'The captioning API returned invalid JSON.\n'
                        f'{response_body}') from exception
                    
            except error.HTTPError as exception:
                # Retry on rate limit (429) and server busy/overloaded (503, 502, 504) errors
                if exception.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                    # Get suggested retry delay from header if available
                    retry_after = exception.headers.get('Retry-After')
                    if retry_after:
                        try:
                            delay = int(retry_after)
                        except ValueError:
                            delay = base_delay * (2 ** attempt)
                    else:
                        delay = base_delay * (2 ** attempt)
                    
                    print(f"Server busy (HTTP {exception.code}), retrying in {delay} seconds... "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                
                # For other HTTP errors or if retries exhausted
                response_body = exception.read().decode('utf-8', errors='replace')
                raise RuntimeError(
                    f'HTTP {exception.code} from captioning API:\n'
                    f'{response_body}') from exception
                    
            except error.URLError as exception:
                # Retry on connection errors
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Connection error, retrying in {delay} seconds... "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                
                raise RuntimeError(
                    f'Could not reach the captioning API at '
                    f'{self.get_chat_completion_url()}.\n{exception.reason}'
                ) from exception

    @staticmethod
    def extract_text_from_content(content: str | list | None) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ''
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get('type') in ('text', 'output_text'):
                text = item.get('text')
                if isinstance(text, str):
                    text_parts.append(text)
        return ''.join(text_parts)

    def get_caption_from_response(self, response: dict) -> str:
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                'The captioning API response did not include any choices.')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError('The captioning API response was malformed.')
        message = first_choice.get('message')
        if isinstance(message, dict):
            content = message.get('content')
            caption = self.extract_text_from_content(content).strip()
            if caption:
                return caption
        text = first_choice.get('text')
        if isinstance(text, str) and text.strip():
            return text.strip()
        raise RuntimeError(
            'The captioning API response did not include any text content.')

    def postprocess_caption(self, caption: str) -> str:
        caption = caption.strip()
        caption_start = self.caption_start.strip()
        if caption_start and not caption.startswith(caption_start):
            caption = f'{caption_start} {caption}'.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption

    def generate_caption(self, image: Image, image_prompt: str) -> tuple[str, str]:
        payload = self.get_request_payload(image_prompt, image)
        response = self.make_request(payload)
        caption = self.get_caption_from_response(response)
        caption = self.postprocess_caption(caption)
        console_output_caption = caption
        return caption, console_output_caption
