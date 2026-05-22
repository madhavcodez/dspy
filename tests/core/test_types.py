import pydantic
import pytest

import dspy
from dspy.core.openai_format import history_entry_to_openai_dict, message_from_openai, response_to_openai_outputs
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMHistoryEntry,
    LMMessage,
    LMOutput,
    LMOutputBuilder,
    LMRefusalPart,
    LMRequest,
    LMRequestPatch,
    LMResponse,
    LMStreamDeltaEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolResultPart,
    LMVideoPart,
)


def _history_entry(message: LMMessage) -> LMHistoryEntry:
    return LMHistoryEntry(
        request=LMRequest(model="model", messages=[message]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )


def test_assistant_message_normalizes_openai_response_fields():
    message = message_from_openai(
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "I cannot help."},
                {"type": "refusal", "refusal": "blocked"},
            ],
            "refusal": "policy",
            "annotations": [],
            "audio": {"id": "audio_1"},
        }
    )

    assert message.text == "I cannot help."
    assert [part.text for part in message.parts if isinstance(part, LMRefusalPart)] == ["blocked", "policy"]


def test_core_message_rejects_openai_content_without_adapter():
    with pytest.raises(pydantic.ValidationError):
        LMMessage(
            role="assistant",
            content=[
                {"type": "output_text", "text": "I cannot help."},
                {"type": "refusal", "refusal": "blocked"},
            ],
        )


def test_assistant_message_normalizes_litellm_provider_fields():
    message = message_from_openai(
        {
            "role": "assistant",
            "content": "I cannot help.",
            "tool_calls": None,
            "function_call": None,
            "provider_specific_fields": {"refusal": "policy", "trace_id": "trace_1"},
            "annotations": [{"type": "url_citation", "url": "https://example.com"}],
            "audio": {"id": "audio_1"},
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "reasoning_content": "policy reasoning",
        }
    )

    assert message.text == "I cannot help."
    assert [part.text for part in message.parts if isinstance(part, LMRefusalPart)] == ["policy"]
    assert [part.text for part in message.parts if isinstance(part, LMThinkingPart)] == ["policy reasoning"]
    assert message.metadata["provider_message"] == {
        "provider_specific_fields": {"refusal": "policy", "trace_id": "trace_1"},
        "audio": {"id": "audio_1"},
        "annotations": [{"type": "url_citation", "url": "https://example.com"}],
        "id": "msg_1",
        "status": "completed",
        "type": "message",
    }


def test_assistant_message_normalizes_legacy_function_call():
    message = message_from_openai(
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "search", "arguments": '{"query": "dspy"}'},
        }
    )

    assert message.parts == [LMToolCallPart(name="search", args={"query": "dspy"})]


def test_tool_result_content_none_normalizes_to_empty_parts():
    assert LMToolResultPart(content=None).content == []

    message = message_from_openai({"role": "tool", "content": None, "tool_call_id": "call_1", "name": "search"})
    result = message.parts[0]

    assert isinstance(result, LMToolResultPart)
    assert result.call_id == "call_1"
    assert result.name == "search"
    assert result.content == []


def test_audio_content_accepts_url_and_serializes_without_data_key():
    message = message_from_openai(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"url": "https://example.com/audio.wav", "format": "wav"},
                }
            ],
        }
    )

    audio = message.parts[0]
    assert isinstance(audio, LMAudioPart)
    assert audio.url == "https://example.com/audio.wav"

    entry = history_entry_to_openai_dict(_history_entry(message))
    input_audio = entry["messages"][0]["content"][0]["input_audio"]

    assert input_audio == {"format": "wav", "url": "https://example.com/audio.wav"}


def test_audio_content_defaults_null_format_to_wav():
    message = message_from_openai(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"url": "https://example.com/audio.wav", "format": None},
                }
            ],
        }
    )

    audio = message.parts[0]

    assert isinstance(audio, LMAudioPart)
    assert audio.media_type == "audio/wav"


def test_image_content_requires_mapping_with_url():
    with pytest.raises(TypeError, match="Image content block"):
        message_from_openai(
            {"role": "user", "content": [{"type": "image_url", "image_url": "https://example.com/image.png"}]}
        )

    with pytest.raises(ValueError, match="requires url"):
        message_from_openai({"role": "user", "content": [{"type": "image_url", "image_url": {}}]})


def test_audio_data_serializes_as_base64_payload_not_data_uri():
    message = dspy.User(LMAudioPart(data="YWJj", media_type="audio/wav"))
    entry = history_entry_to_openai_dict(_history_entry(message))

    input_audio = entry["messages"][0]["content"][0]["input_audio"]

    assert input_audio == {"format": "wav", "data": "YWJj"}


def test_binary_part_serializes_as_openai_file_content_and_round_trips():
    message = dspy.User(LMBinaryPart(file_id="file_123", filename="report.pdf"))
    entry = history_entry_to_openai_dict(_history_entry(message))

    content = entry["messages"][0]["content"][0]
    round_tripped = message_from_openai(entry["messages"][0]).parts[0]

    assert content == {
        "type": "file",
        "file": {
            "media_type": "application/octet-stream",
            "file_id": "file_123",
            "filename": "report.pdf",
        },
    }
    assert isinstance(round_tripped, LMBinaryPart)
    assert round_tripped.file_id == "file_123"
    assert round_tripped.filename == "report.pdf"


def test_video_data_round_trips_through_history_messages():
    message = dspy.User(LMVideoPart(data="YWJj", media_type="video/mp4"))
    entry = history_entry_to_openai_dict(_history_entry(message))

    content = entry["messages"][0]["content"][0]
    round_tripped = message_from_openai(entry["messages"][0]).parts[0]

    assert content == {
        "type": "video",
        "video": {"media_type": "video/mp4", "data": "data:video/mp4;base64,YWJj"},
    }
    assert isinstance(round_tripped, LMVideoPart)
    assert round_tripped.data == "YWJj"
    assert round_tripped.url is None


def test_video_file_id_round_trips_through_history_messages():
    message = dspy.User(LMVideoPart(file_id="file_video", media_type="video/mp4"))
    entry = history_entry_to_openai_dict(_history_entry(message))

    round_tripped = message_from_openai(entry["messages"][0]).parts[0]

    assert isinstance(round_tripped, LMVideoPart)
    assert round_tripped.file_id == "file_video"
    assert round_tripped.url is None


def test_document_source_url_stays_url_and_round_trips_through_history_messages():
    message = message_from_openai(
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": "https://example.com/report.pdf",
                    "title": "Report",
                }
            ],
        }
    )
    document = message.parts[0]
    assert isinstance(document, LMDocumentPart)
    assert document.url == "https://example.com/report.pdf"
    assert document.data is None

    entry = history_entry_to_openai_dict(_history_entry(message))
    round_tripped = message_from_openai(entry["messages"][0]).parts[0]

    assert isinstance(round_tripped, LMDocumentPart)
    assert round_tripped.url == "https://example.com/report.pdf"
    assert round_tripped.data is None


def test_config_extensions_flatten_when_converted_to_legacy_kwargs():
    config = dspy.LMConfig.from_kwargs(temperature=0.2, provider_flag=True)
    patch = LMRequestPatch(config=config)
    request = LMRequest.from_call(model="model", prompt="hi", temperature=0.2, provider_flag=True)
    entry = LMHistoryEntry(
        request=request,
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

    assert patch.as_lm_kwargs() == {"provider_flag": True, "temperature": 0.2}
    assert history_entry_to_openai_dict(entry)["kwargs"] == {"provider_flag": True, "temperature": 0.2}


def test_reasoning_effort_overrides_nested_reasoning_config():
    config = dspy.LMConfig.from_kwargs(reasoning={"effort": "low", "summary": "auto"}, reasoning_effort="high")

    assert config.reasoning.effort == "high"
    assert config.reasoning.summary == "auto"


def test_default_config_does_not_serialize_empty_stop_sequences():
    request = LMRequest.from_call(model="model", prompt="hi")
    entry = LMHistoryEntry(
        request=request,
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

    assert request.config.stop is None
    assert history_entry_to_openai_dict(entry)["kwargs"] == {}


def test_history_entry_exposes_typed_convenience_properties():
    message = dspy.User("hi")
    request = LMRequest.from_call(model="model", messages=[message], temperature=0.2)
    response = LMResponse.from_text("ok", model="response-model", usage={"input_tokens": 1}, cost=0.5)
    entry = LMHistoryEntry(request=request, response=response, timestamp="timestamp", uuid="uuid")

    assert entry.model == "model"
    assert entry.prompt == "hi"
    assert entry.messages == [message]
    assert entry.outputs == response.outputs
    assert entry.usage.input_tokens == 1
    assert entry.cost == 0.5
    assert entry.config == request.config
    assert entry.tools == []
    assert entry.response_model == "response-model"
    assert entry["messages"] == [message]
    assert entry["config"] == request.config
    assert "kwargs" not in entry


def test_response_rejects_empty_outputs():
    with pytest.raises(pydantic.ValidationError):
        LMResponse(model="model", outputs=[])


def test_response_exposes_refusal_from_first_output():
    response = LMResponse(model="model", outputs=[LMOutput(parts=[LMRefusalPart(text="blocked")])])

    assert response.refusal == "blocked"


def test_output_to_value_preserves_redacted_thinking_part():
    thinking = LMThinkingPart(text="hidden", redacted=True)
    output = LMOutput(parts=[thinking])

    assert output.to_value() == [thinking]


def test_response_to_openai_outputs_keeps_refusal_separate_from_text():
    response = LMResponse(model="model", outputs=[LMOutput(parts=[LMRefusalPart(text="blocked")])])

    assert response_to_openai_outputs(response) == [{"text": None, "refusal": "blocked"}]


def test_stream_event_indices_must_be_non_negative():
    with pytest.raises(pydantic.ValidationError):
        LMStreamDeltaEvent(output_index=-1, part_index=0, delta=LMTextDelta(text="x"))

    with pytest.raises(pydantic.ValidationError):
        LMStreamDeltaEvent(output_index=0, part_index=-1, delta=LMTextDelta(text="x"))


def test_stream_builder_rejects_sparse_output_indices():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=2, part_index=0, delta=LMTextDelta(text="third")))

    with pytest.raises(ValueError, match="output indices"):
        builder.to_response()


def test_stream_builder_rejects_sparse_part_indices():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=0, part_index=1, delta=LMTextDelta(text="second")))

    with pytest.raises(ValueError, match="part indices"):
        builder.to_response()


def test_stream_builder_rejects_delta_type_changes():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMTextDelta(text="text")))

    with pytest.raises(ValueError, match="thinking delta"):
        builder.apply(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMThinkingDelta(text="thought")))


def test_stream_builder_rejects_incomplete_tool_call_arguments():
    builder = LMOutputBuilder()
    builder.apply(
        LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=LMToolCallDelta(id="call_1", name="search", args_delta='{"query": '),
        )
    )

    with pytest.raises(ValueError, match="tool-call arguments"):
        builder.to_response()


def test_public_core_types_are_exported_from_dspy():
    assert dspy.LMRequest is LMRequest
    assert dspy.System("hello").role == "system"
