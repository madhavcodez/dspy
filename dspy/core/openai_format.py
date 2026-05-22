"""OpenAI/LiteLLM wire-format adapters for DSPy core LM types."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import pydantic

from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMConfig,
    LMDocumentPart,
    LMHistoryEntry,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMPart,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolResultPart,
    LMVideoPart,
)

_SOURCE_KEYS = ("data", "url", "file_id", "path")
_PROVIDER_MESSAGE_FIELDS = ("audio", "annotations", "id", "status", "type")


def message_from_openai(message: Mapping[str, Any] | LMMessage) -> LMMessage:
    """Convert an OpenAI/LiteLLM message dictionary into an `LMMessage`."""
    if isinstance(message, LMMessage):
        return message
    if not isinstance(message, Mapping):
        raise TypeError("OpenAI message must be a mapping.")

    data = dict(message)
    refusal = data.pop("refusal", None)
    function_call = data.pop("function_call", None)
    reasoning_content = data.pop("reasoning_content", None)
    provider_specific_fields = data.pop("provider_specific_fields", None)
    if isinstance(provider_specific_fields, Mapping):
        if refusal is None:
            refusal = provider_specific_fields.get("refusal")
        _store_provider_message_field(data, "provider_specific_fields", provider_specific_fields)
    elif provider_specific_fields is not None:
        _store_provider_message_field(data, "provider_specific_fields", provider_specific_fields)

    for key in _PROVIDER_MESSAGE_FIELDS:
        if key in data:
            _store_provider_message_field(data, key, data.pop(key))

    if data.get("role") == "tool" and "parts" not in data:
        content = data.pop("content", None)
        call_id = data.pop("tool_call_id", None)
        name = data.pop("name", None)
        data["parts"] = [
            LMToolResultPart(
                call_id=call_id,
                name=name,
                content=parts_from_openai_content(content),
            )
        ]
    else:
        provider_parts = _openai_message_provider_parts(
            refusal=refusal,
            function_call=function_call,
            reasoning_content=reasoning_content,
            tool_calls=data.pop("tool_calls", None),
        )
        if "parts" not in data:
            data["parts"] = parts_from_openai_content(data.pop("content", None)) if "content" in data else []
        else:
            data["parts"] = [_part_from_value(part) for part in data["parts"]]
        data["parts"].extend(provider_parts)

    return LMMessage(**data)


def messages_from_openai(messages: Sequence[Mapping[str, Any] | LMMessage]) -> list[LMMessage]:
    return [message_from_openai(message) for message in messages]


def message_to_openai(message: LMMessage) -> dict[str, Any]:
    if message.role == "assistant":
        tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
        content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
        item: dict[str, Any] = {
            "role": "assistant",
            "content": _message_parts_to_openai_content(content_parts) if content_parts else None,
        }
        if tool_calls:
            item["tool_calls"] = [tool_call_to_provider_dict(call) for call in tool_calls]
    elif message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
        result = message.parts[0]
        item = {"role": "tool", "content": _tool_result_content(result)}
        if result.call_id is not None:
            item["tool_call_id"] = result.call_id
        if result.name is not None:
            item["name"] = result.name
    else:
        item = {
            "role": message.role,
            "content": _message_parts_to_openai_content(message.parts),
        }
    if message.name is not None and "name" not in item:
        item["name"] = message.name
    return item


def messages_to_openai(messages: Sequence[LMMessage]) -> list[dict[str, Any]]:
    return [message_to_openai(message) for message in messages]


def request_to_openai_prompt(request: LMRequest) -> str | None:
    if len(request.messages) != 1:
        return None
    message = request.messages[0]
    if message.role != "user" or len(message.parts) != 1:
        return None
    part = message.parts[0]
    return part.text if isinstance(part, LMTextPart) else None


def request_to_openai_messages(request: LMRequest) -> list[dict[str, Any]]:
    return messages_to_openai(request.messages)


def request_to_openai_kwargs(request: LMRequest) -> dict[str, Any]:
    return lm_config_to_openai_kwargs(request.config)


def lm_config_to_openai_kwargs(config: LMConfig) -> dict[str, Any]:
    data = config.model_dump(exclude_none=True)
    extensions = data.pop("extensions", {}) or {}
    return {**extensions, **data}


def output_to_openai_dict(output: LMOutput) -> dict[str, Any]:
    data: dict[str, Any] = {"text": output.text}
    if output.refusal is not None:
        data["refusal"] = output.refusal
    if output.reasoning_content is not None:
        data["reasoning_content"] = output.reasoning_content
    if output.tool_calls:
        data["tool_calls"] = [tool_call_to_provider_dict(call) for call in output.tool_calls]
    if output.citations:
        data["citations"] = [citation.model_dump(exclude_none=True) for citation in output.citations]
    if output.logprobs is not None:
        data["logprobs"] = output.logprobs
    return data


def response_to_openai_outputs(response: LMResponse) -> list[Any]:
    outputs: list[Any] = []
    for output in response.outputs:
        if _requires_output_dict(output):
            outputs.append(output_to_openai_dict(output))
        else:
            outputs.append(output.to_value())
    return outputs


def history_entry_to_openai_dict(entry: LMHistoryEntry) -> dict[str, Any]:
    data = entry.model_dump(mode="python", exclude_none=True)
    data.update(entry.model_extra or {})
    data.update(
        {
            "outputs": response_to_openai_outputs(entry.response),
            "usage": entry.response.usage_as_dict(),
            "cost": entry.response.cost,
            "model": entry.request.model,
            "prompt": request_to_openai_prompt(entry.request),
            "messages": request_to_openai_messages(entry.request),
            "kwargs": request_to_openai_kwargs(entry.request),
            "response_model": entry.response.model,
        }
    )
    return {key: value for key, value in data.items() if value is not None}


def parts_from_openai_content(content: Any) -> list[LMPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [_part_from_value(content)]

    parts = []
    for item in content:
        item_type = item.get("type") if isinstance(item, Mapping) else None
        if item_type == "text":
            parts.append(LMTextPart(text=item.get("text", "")))
        elif item_type == "output_text":
            parts.append(LMTextPart(text=item.get("text", "")))
        elif item_type == "refusal":
            parts.append(LMRefusalPart(text=item.get("refusal", item.get("text", ""))))
        elif item_type == "image_url":
            parts.append(_image_dict_to_part(item.get("image_url", {})))
        elif item_type == "input_audio":
            parts.append(_audio_dict_to_part(item.get("input_audio", {})))
        elif item_type == "file":
            parts.append(_binary_dict_to_part(item.get("file", {})))
        elif item_type == "document":
            parts.append(_document_dict_to_part(item))
        elif item_type == "video":
            video = item.get("video", {})
            parts.append(_media_dict_to_video_part(video))
        else:
            parts.append(_part_from_value(item))
    return parts


def tool_call_from_openai(tool_call: Any) -> LMToolCallPart:
    if isinstance(tool_call, LMToolCallPart):
        return tool_call
    if not isinstance(tool_call, Mapping):
        raise TypeError(f"Cannot convert {type(tool_call)!r} to an LMToolCallPart.")

    function = tool_call.get("function", {})
    if not isinstance(function, Mapping):
        function = {}

    args = function.get("arguments", {})
    if isinstance(args, str):
        args = _parse_json_object(args)
    elif isinstance(args, Mapping):
        args = dict(args)
    else:
        args = {}

    return LMToolCallPart(
        id=tool_call.get("id"),
        name=function.get("name") or tool_call.get("name") or "",
        args=args,
    )


def tool_calls_from_openai(tool_calls: Sequence[Any]) -> list[LMToolCallPart]:
    return [tool_call_from_openai(tool_call) for tool_call in tool_calls]


def tool_call_to_provider_dict(call: LMToolCallPart) -> dict[str, Any]:
    data = {
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args),
        },
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _store_provider_message_field(data: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return
    raw_metadata = data.get("metadata")
    if raw_metadata is None:
        metadata = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        return
    raw_provider_message = metadata.get("provider_message")
    provider_message = dict(raw_provider_message) if isinstance(raw_provider_message, Mapping) else {}
    provider_message[key] = value
    metadata["provider_message"] = provider_message
    data["metadata"] = metadata


def _openai_message_provider_parts(
    *,
    refusal: Any,
    function_call: Any,
    reasoning_content: Any,
    tool_calls: Any,
) -> list[LMPart]:
    parts: list[LMPart] = []
    if refusal is not None:
        parts.append(LMRefusalPart(text=refusal))
    if function_call is not None:
        parts.append(tool_call_from_openai({"function": function_call}))
    if reasoning_content is not None:
        parts.append(LMThinkingPart(text=reasoning_content))
    if tool_calls:
        parts.extend(tool_calls_from_openai(tool_calls))
    return parts


def _message_parts_to_openai_content(parts: list[LMPart]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], LMTextPart):
        return parts[0].text
    return [_part_to_openai_content(part) for part in parts]


def _part_to_openai_content(part: LMPart) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        return {"type": "image_url", "image_url": {"url": _part_source(part)}}
    if isinstance(part, LMAudioPart):
        return {
            "type": "input_audio",
            "input_audio": _audio_source(part),
        }
    if isinstance(part, LMVideoPart):
        return {"type": "video", "video": _video_source(part)}
    if isinstance(part, LMDocumentPart):
        data = {"type": "document"}
        if part.source is not None:
            data["source"] = part.source
        else:
            data["source"] = _part_source(part)
            data["media_type"] = part.media_type
        if part.citations:
            data["citations"] = part.citations
        if part.title is not None:
            data["title"] = part.title
        if part.context is not None:
            data["context"] = part.context
        return data
    if isinstance(part, LMBinaryPart):
        return {"type": "file", "file": _binary_file(part)}
    return part.model_dump(exclude_none=True)


def _tool_result_content(result: LMToolResultPart) -> str:
    chunks = []
    for part in result.content:
        if isinstance(part, LMTextPart):
            chunks.append(part.text)
        else:
            chunks.append(json.dumps(part.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
    return "".join(chunks)


def _part_source(part: LMImagePart | LMAudioPart | LMVideoPart | LMDocumentPart | LMBinaryPart) -> str | None:
    if part.data is not None:
        return _data_uri(part.data, part.media_type)
    return part.url or part.file_id or part.path


def _media_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type


def _data_uri(data: str, media_type: str) -> str:
    return data if data.startswith("data:") else f"data:{media_type};base64,{data}"


def _audio_source(part: LMAudioPart) -> dict[str, Any]:
    data: dict[str, Any] = {"format": _media_format(part.media_type)}
    if part.data is not None:
        if part.data.startswith("data:"):
            media_type, audio_data = _split_data_uri(part.data)
            data["format"] = _media_format(media_type)
            data["data"] = audio_data
        else:
            data["data"] = part.data
    elif part.url is not None:
        data["url"] = part.url
    elif part.file_id is not None:
        data["file_id"] = part.file_id
    elif part.path is not None:
        data["path"] = part.path
    return data


def _binary_file(part: LMBinaryPart) -> dict[str, Any]:
    payload = {"media_type": part.media_type}
    if part.data is not None:
        payload["file_data"] = _data_uri(part.data, part.media_type)
    elif part.url is not None:
        payload["url"] = part.url
    elif part.file_id is not None:
        payload["file_id"] = part.file_id
    elif part.path is not None:
        payload["path"] = part.path
    if part.filename is not None:
        payload["filename"] = part.filename
    return payload


def _video_source(part: LMVideoPart) -> dict[str, Any]:
    video = {"media_type": part.media_type}
    if part.data is not None:
        video["data"] = _data_uri(part.data, part.media_type)
    elif part.url is not None:
        video["url"] = part.url
    elif part.file_id is not None:
        video["file_id"] = part.file_id
    elif part.path is not None:
        video["path"] = part.path
    return video


def _part_from_value(value: Any) -> LMPart:
    if isinstance(
        value,
        (
            LMTextPart,
            LMImagePart,
            LMAudioPart,
            LMVideoPart,
            LMDocumentPart,
            LMBinaryPart,
            LMToolCallPart,
            LMToolResultPart,
            LMThinkingPart,
            LMCitationPart,
            LMRefusalPart,
        ),
    ):
        return value
    if isinstance(value, str):
        return LMTextPart(text=value)
    if isinstance(value, Mapping) and "type" in value:
        return pydantic.TypeAdapter(LMPart).validate_python(value)
    raise TypeError(f"Cannot convert {type(value)!r} to an LMPart.")


def _image_source_to_part(source: str) -> LMImagePart:
    if not isinstance(source, str):
        raise TypeError("Image URL must be a string.")
    if source.startswith("data:"):
        media_type, data = _split_data_uri(source)
        return LMImagePart(data=data, media_type=media_type)
    media_type = mimetypes.guess_type(urlparse(source).path)[0] or "image/png"
    return LMImagePart(url=source, media_type=media_type)


def _image_dict_to_part(image: Mapping[str, Any]) -> LMImagePart:
    if not isinstance(image, Mapping):
        raise TypeError("Image content block must be a mapping.")
    url = image.get("url")
    if url is None:
        raise ValueError("Image content block requires url.")
    return _image_source_to_part(url)


def _binary_dict_to_part(file_block: Mapping[str, Any]) -> LMBinaryPart:
    if not isinstance(file_block, Mapping):
        raise TypeError("Binary content block must be a mapping.")

    filename = file_block.get("filename")
    media_type = file_block.get("media_type") or "application/octet-stream"
    if file_block.get("file_data") is not None:
        media_type, data = _split_data_uri(file_block["file_data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=filename)
    if file_block.get("data") is not None:
        media_type, data = _media_data(file_block["data"], default_media_type=media_type)
        return LMBinaryPart(data=data, media_type=media_type, filename=filename)
    if file_block.get("url") is not None:
        return LMBinaryPart(url=file_block["url"], media_type=media_type, filename=filename)
    if file_block.get("file_id") is not None:
        return LMBinaryPart(file_id=file_block["file_id"], media_type=media_type, filename=filename)
    if file_block.get("path") is not None:
        return LMBinaryPart(path=file_block["path"], media_type=media_type, filename=filename)
    raise ValueError("Binary content block requires data, file_data, url, file_id, or path.")


def _audio_dict_to_part(audio: Mapping[str, Any]) -> LMAudioPart:
    if not isinstance(audio, Mapping):
        raise TypeError("Audio content block must be a mapping.")

    audio_format = audio.get("format") or "wav"
    if not isinstance(audio_format, str):
        raise TypeError("Audio format must be a string.")
    media_type = audio_format if "/" in audio_format else f"audio/{audio_format}"
    if audio.get("data") is not None:
        data = audio["data"]
        if isinstance(data, str) and data.startswith("data:"):
            media_type, data = _split_data_uri(data)
        return LMAudioPart(data=data, media_type=media_type)
    if audio.get("url") is not None:
        return LMAudioPart(url=audio["url"], media_type=media_type)
    if audio.get("file_id") is not None:
        return LMAudioPart(file_id=audio["file_id"], media_type=media_type)
    if audio.get("path") is not None:
        return LMAudioPart(path=audio["path"], media_type=media_type)
    raise ValueError("Audio content block requires data, url, file_id, or path.")


def _document_dict_to_part(item: Mapping[str, Any]) -> LMDocumentPart:
    common = {"title": item.get("title"), "context": item.get("context")}
    media_type = item.get("media_type") or "application/pdf"
    for source_key in _SOURCE_KEYS:
        if item.get(source_key) is not None:
            return LMDocumentPart(**{source_key: item[source_key]}, media_type=media_type, **common)

    source = item.get("source")
    if isinstance(source, dict):
        return LMDocumentPart(
            source=source,
            citations=item.get("citations") or {},
            **common,
        )
    if isinstance(source, str):
        kwargs = _media_source_kwargs(source, default_media_type=media_type)
        return LMDocumentPart(**kwargs, **common)
    raise ValueError("Document content block requires source.")


def _media_dict_to_video_part(video: Mapping[str, Any]) -> LMVideoPart:
    if not isinstance(video, Mapping):
        raise TypeError("Video content block must be a mapping.")

    media_type = video.get("media_type") or "video/mp4"
    if video.get("data") is not None:
        media_type, data = _media_data(video["data"], default_media_type=media_type)
        return LMVideoPart(data=data, media_type=media_type)
    if video.get("url") is not None:
        if isinstance(video["url"], str) and video["url"].startswith("data:"):
            media_type, data = _split_data_uri(video["url"])
            return LMVideoPart(data=data, media_type=media_type)
        return LMVideoPart(url=video["url"], media_type=media_type)
    if video.get("file_id") is not None:
        return LMVideoPart(file_id=video["file_id"], media_type=media_type)
    if video.get("path") is not None:
        return LMVideoPart(path=video["path"], media_type=media_type)
    raise ValueError("Video content block requires data, url, file_id, or path.")


def _split_data_uri(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise TypeError("Media data must be a string.")
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def _media_data(value: str, *, default_media_type: str) -> tuple[str, str]:
    media_type, data = _split_data_uri(value)
    if value.startswith("data:"):
        return media_type, data
    return default_media_type, data


def _media_source_kwargs(source: str, *, default_media_type: str) -> dict[str, str]:
    if source.startswith("data:"):
        media_type, data = _split_data_uri(source)
        return {"data": data, "media_type": media_type}

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        media_type = mimetypes.guess_type(parsed.path)[0] or default_media_type
        return {"url": source, "media_type": media_type}

    return {"file_id": source, "media_type": default_media_type}


def _requires_output_dict(output: LMOutput) -> bool:
    return bool(
        output.logprobs is not None
        or output.refusal is not None
        or output.reasoning_content is not None
        or output.tool_calls
        or output.citations
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
