from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any

import torch

logger = logging.getLogger(__name__)


class InferenceBatcher:

    def __init__(
        self,
        model: Any,
        pad_token_id: int,
        max_batch_size: int = 10,
        max_wait_ms: float = 50,
    ) -> None:
        self._model = model
        self._pad_token_id = pad_token_id
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: queue.Queue[tuple[torch.Tensor, dict, Future] | None] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, input_ids: torch.Tensor, gen_kwargs: dict[str, Any]) -> torch.Tensor:
        """Submit a single inference request and block until the result is ready.

        Parameters
        ----------
        input_ids:
            1-D tensor of token ids (already on the model's device).
        gen_kwargs:
            Generation keyword arguments (``max_new_tokens``, ``temperature``, etc.).

        Returns
        -------
        torch.Tensor
            1-D tensor of newly generated token ids (on CPU).
        """
        future: Future[torch.Tensor] = Future()
        self._queue.put((input_ids, gen_kwargs, future))
        return future.result()

    def shutdown(self) -> None:
        """Signal the worker thread to exit."""
        self._queue.put(None)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Continuously drain the queue, batch requests, and run inference."""
        while True:
            batch: list[tuple[torch.Tensor, dict, Future]] = []

            # Block until at least one request arrives
            first = self._queue.get()
            if first is None:
                return  # shutdown sentinel
            batch.append(first)

            # Collect more requests up to max_batch_size or deadline
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    if item is None:
                        # Put sentinel back so next iteration exits too
                        self._queue.put(None)
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            # Run the batch
            try:
                self._run_batch(batch)
            except Exception as exc:
                # Propagate the exception to all waiting futures
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    def _run_batch(self, batch: list[tuple[torch.Tensor, dict, Future]]) -> None:
        """Left-pad, run model.generate(), dispatch results."""
        all_input_ids = [item[0] for item in batch]
        gen_kwargs = dict(batch[0][1])  # use first request's kwargs
        futures = [item[2] for item in batch]

        device = all_input_ids[0].device
        max_len = max(ids.shape[0] for ids in all_input_ids)

        # Left-pad all sequences to the same length
        padded_ids = []
        attention_masks = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len,), self._pad_token_id, dtype=ids.dtype, device=device,
                )
                padded_ids.append(torch.cat([padding, ids]))
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(ids.shape[0], dtype=torch.long, device=device),
                ])
            else:
                padded_ids.append(ids)
                mask = torch.ones(ids.shape[0], dtype=torch.long, device=device)
            attention_masks.append(mask)

        batch_input = torch.stack(padded_ids)
        batch_mask = torch.stack(attention_masks)

        logger.info("[batcher] Running batch of %d requests (max_len=%d)", len(batch), max_len)

        with torch.no_grad():
            outputs = self._model.generate(
                batch_input,
                attention_mask=batch_mask,
                **gen_kwargs,
            )

        # Dispatch results: extract only the newly generated tokens for each sequence
        for i, future in enumerate(futures):
            new_tokens = outputs[i, max_len:]
            # Strip trailing pad tokens
            mask = new_tokens != self._pad_token_id
            if mask.any():
                last_real = mask.nonzero()[-1].item() + 1
                new_tokens = new_tokens[:last_real]
            future.set_result(new_tokens.cpu())
