import threading
from typing import Any, ClassVar

from app.config import Settings


class GenerationError(RuntimeError):
    """Raised when the configured language model cannot produce an answer."""


class AnswerGenerator:
    _pipelines: ClassVar[dict[tuple[str, int], Any]] = {}
    _pipeline_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _load_pipeline(self) -> Any:
        cache_key = (self.settings.generation_model, self.settings.generation_device)
        with self._pipeline_lock:
            if cache_key not in self._pipelines:
                try:
                    from transformers import pipeline

                    self._pipelines[cache_key] = pipeline(
                        "text2text-generation",
                        model=self.settings.generation_model,
                        device=self.settings.generation_device,
                    )
                except Exception as error:
                    raise GenerationError(
                        "Could not load the local Hugging Face generation model. "
                        "Check the model name, local cache, and network access for its first download."
                    ) from None
            return self._pipelines[cache_key]

    @staticmethod
    def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
        context = "\n\n".join(chunk["text"] for chunk in chunks)
        return (
            "Use only the context to answer. If the answer is not stated, say it is not available in the uploaded documents. "
            "Do not invent facts.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            raise GenerationError("Answer generation requires at least one retrieved source.")
        generator = self._load_pipeline()
        try:
            response = generator(
                self.build_prompt(question, chunks),
                max_new_tokens=self.settings.generation_max_new_tokens,
                do_sample=False,
                truncation=True,
            )
            answer = response[0].get("generated_text", "") if response else ""
            if not answer or not answer.strip():
                raise GenerationError("The local generation model returned an empty answer.")
            return answer.strip()
        except GenerationError:
            raise
        except Exception as error:
            raise GenerationError(
                f"Local Hugging Face generation failed ({type(error).__name__})."
            ) from None
