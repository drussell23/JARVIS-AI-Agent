"""
Centralized Embedding Service v1.0 - Enterprise-Grade Embedding Management
============================================================================

This module provides a SINGLE, CENTRALIZED SentenceTransformer instance that is
shared across the entire JARVIS codebase. This eliminates the semaphore leak
caused by multiple SentenceTransformer instances creating independent
torch.multiprocessing pools.

ROOT CAUSE FIX:
    Previously, SentenceTransformer was instantiated in 14+ different modules:
    - backend/ml_model_loader.py
    - backend/core/trinity_knowledge_graph.py
    - backend/intelligence/long_term_memory.py
    - backend/neural_mesh/knowledge/semantic_memory.py
    - ... and more

    Each instance could spawn internal multiprocessing pools for parallel encoding.
    These pools create semaphores that weren't being cleaned up, causing:
    "resource_tracker: There appear to be 1 leaked semaphore objects to clean up"

SOLUTION:
    1. Single SentenceTransformer instance managed by this service
    2. Lazy loading - model only loaded when first needed
    3. Proper cleanup via stop_multi_process_pool() if pools were started
    4. Thread-safe access with connection pooling semantics
    5. Registered with GracefulShutdown for proper cleanup order

Usage:
    from backend.core.embedding_service import get_embedding_service

    # Get the shared service (lazy-loads model on first call)
    service = await get_embedding_service()

    # Generate embeddings
    embeddings = await service.encode(["text1", "text2"])

    # Or use the convenience function
    from backend.core.embedding_service import encode_texts
    embeddings = await encode_texts(["text1", "text2"])

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import atexit
import gc
import logging
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Adaptive Model Tiering — the Sovereign Adaptive Memory Matrix (Slice 259)
# =============================================================================
class EmbeddingTier(IntEnum):
    """Fidelity-ordered embedding tiers (higher value = higher fidelity).

    The service runs on exactly one tier at a time and adapts to host memory:

      * ``HIGH``  — the ~800MB PyTorch SentenceTransformer (best quality).
      * ``LITE``  — the ~200MB fastembed (ONNX/CoreML, ARM64-optimized) model,
                    embedding-dimension-compatible (384-dim bge-small ≈
                    all-MiniLM-L6-v2). The tier the service *demotes* to when
                    the HIGH allocation is denied under memory pressure.
      * ``NONE``  — no model loaded yet (or fully degraded).

    Ordering matters: ``maybe_promote_tier`` only ever moves *up* (LITE→HIGH),
    and the watchdog never demotes a working HIGH tier.
    """

    NONE = 0
    LITE = 1
    HIGH = 2

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingServiceConfig:
    """Configuration for the embedding service."""

    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu", "cuda", "mps"
    # Slice 153 — fastembed fallback model (used when sentence-transformers is
    # unavailable, e.g. the Oracle-capable soak image which ships fastembed not
    # torch). 384-dim like all-MiniLM-L6-v2 → embedding-dim compatible. The model
    # name is config-sourced (no hardcode in logic); matches semantic_index's default.
    fastembed_model_name: str = "BAAI/bge-small-en-v1.5"

    # Performance settings
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = False

    # Multi-process pool settings (disabled by default to prevent leaks)
    use_multiprocess_pool: bool = False
    pool_size: int = 0  # 0 = no pool

    # Cache settings
    enable_cache: bool = True
    cache_maxsize: int = 10000

    # Type hints for cache (key is string hash)
    _cache_key_type: str = "str"  # Document that cache keys are strings

    # Timeouts
    encode_timeout: float = 30.0
    shutdown_timeout: float = 10.0

    # ── Adaptive Model Tiering (Slice 259) ──────────────────────────────
    # Master switch for the demote-on-pressure / promote-on-headroom engine.
    adaptive_tiering_enabled: bool = True
    # Memory the two tiers are expected to cost (MB). Drives the budget gate +
    # the promotion-headroom check. Env-tunable — no hardcoded thresholds in
    # the logic.
    pytorch_estimate_mb: int = 800
    fastembed_estimate_mb: int = 200
    # Background promotion poller (LITE→HIGH).
    promotion_enabled: bool = True
    promotion_poll_s: float = 60.0
    # Hysteresis: require this many *consecutive* headroom observations before
    # committing to an 800MB promotion (stops flapping when memory hovers near
    # the line).
    promotion_stable_checks: int = 2
    # Extra free GB (beyond the model estimate) required before promoting —
    # the floor the host must keep after the HIGH model loads.
    promotion_headroom_gb: float = 2.0
    # Free GB (beyond the LITE estimate) required to even attempt the LITE
    # tier; below this the service degrades to no embeddings.
    lite_floor_gb: float = 0.5

    @classmethod
    def from_env(cls) -> "EmbeddingServiceConfig":
        """Create config from environment variables."""
        def _flag(name: str, default: bool) -> bool:
            return os.getenv(name, str(default)).strip().lower() in (
                "1", "true", "yes", "on",
            )
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            normalize_embeddings=os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
            use_multiprocess_pool=os.getenv("EMBEDDING_MULTIPROCESS", "false").lower() == "true",
            pool_size=int(os.getenv("EMBEDDING_POOL_SIZE", "0")),
            enable_cache=os.getenv("EMBEDDING_CACHE", "true").lower() == "true",
            cache_maxsize=int(os.getenv("EMBEDDING_CACHE_SIZE", "10000")),
            fastembed_model_name=os.getenv(
                "EMBEDDING_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5",
            ),
            adaptive_tiering_enabled=_flag("JARVIS_EMBEDDING_ADAPTIVE_TIERING", True),
            pytorch_estimate_mb=int(os.getenv("EMBEDDING_PYTORCH_ESTIMATE_MB", "800")),
            fastembed_estimate_mb=int(os.getenv("EMBEDDING_FASTEMBED_ESTIMATE_MB", "200")),
            promotion_enabled=_flag("JARVIS_EMBEDDING_ADAPTIVE_PROMOTION", True),
            promotion_poll_s=float(os.getenv("EMBEDDING_PROMOTION_POLL_S", "60")),
            promotion_stable_checks=max(
                1, int(os.getenv("EMBEDDING_PROMOTION_STABLE_CHECKS", "2"))
            ),
            promotion_headroom_gb=float(os.getenv("EMBEDDING_PROMOTION_HEADROOM_GB", "2.0")),
            lite_floor_gb=float(os.getenv("EMBEDDING_LITE_FLOOR_GB", "0.5")),
        )


# =============================================================================
# Slice 153 — fastembed fallback adapter (SentenceTransformer.encode-compatible)
# =============================================================================

class _FastembedSTAdapter:
    """A ``SentenceTransformer.encode``-compatible adapter over fastembed's
    ``TextEmbedding``. Slice 153 — lets ``EmbeddingService.encode()`` (and every
    one of its 14+ consumers, incl. the Oracle's ``embed_nodes``) work UNCHANGED on
    fastembed when sentence-transformers is unavailable (the Oracle-capable soak
    image ships fastembed, not torch). Exposes only the ``encode(...)`` subset
    EmbeddingService calls; returns a normalized float32 ``(n, dim)`` array."""

    def __init__(self, model_name: str, *, factory: Any = None) -> None:
        self.model_name = model_name
        if factory is not None:
            self._te = factory(model_name)          # injectable for tests
        else:
            from fastembed import TextEmbedding      # deferred: no import unless used
            self._te = TextEmbedding(model_name=model_name)

    def encode(
        self,
        sentences: Any,
        batch_size: Any = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **_kw: Any,
    ) -> np.ndarray:
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        vecs = [np.asarray(v, dtype="float32") for v in self._te.embed(items)]
        if not vecs:
            return np.zeros((0, 0), dtype="float32")
        arr = np.vstack(vecs).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    """
    Centralized embedding service with proper resource management.

    This is a SINGLETON - only one instance should exist per process.
    Use get_embedding_service() to access it.
    """

    _instance: Optional["EmbeddingService"] = None
    _instance_lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "EmbeddingService":
        """Ensure singleton pattern."""
        del args, kwargs  # Unused but required for signature
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: Optional[EmbeddingServiceConfig] = None):
        """Initialize the embedding service."""
        # Only initialize once
        if self._initialized:
            return

        self._config = config or EmbeddingServiceConfig.from_env()
        self._model = None
        self._model_lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self._pool = None
        self._pool_started = False
        self._shutdown_requested = False
        self._encode_count = 0
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._active_grant = None  # Memory Control Plane grant, if loaded via broker

        # ── Adaptive Model Tiering state (Slice 259) ────────────────────
        self._active_tier: EmbeddingTier = EmbeddingTier.NONE
        self._promotion_task: Optional[asyncio.Task] = None
        self._promotion_stable_count: int = 0
        self._tier_transitions: int = 0  # observability: total demote/promote events
        # The promotion BREAKER. A missing package is not a transient
        # condition: once the HIGH tier's import has failed, every poll
        # would fail the same way for the life of this process. Measured
        # 2026-09-06 (bt-2026-09-06-074921): "Promotion HIGH load failed (No
        # module named 'sentence_transformers')" every two minutes, all
        # session, each one a budget reservation taken and released for an
        # import that cannot succeed. Set once, never cleared; the reason
        # is reported on the observability surface.
        self._promotion_unavailable_reason: Optional[str] = None

        # Register cleanup
        # Guarded: KeyboardInterrupt/SystemExit are BaseExceptions, so the
        # `except Exception` inside these handlers never caught them and a
        # Ctrl+C landing here printed a traceback over the goodbye. Imported
        # locally so this can never introduce an import cycle.
        from backend.core.ouroboros.governance.exit_guard import (
            guarded_atexit_register,
        )
        guarded_atexit_register(self._sync_cleanup)
        self._register_with_shutdown_manager()
        self._register_with_cross_repo_cleanup()

        self._initialized = True
        logger.info(f"[EmbeddingService] Initialized (model: {self._config.model_name})")

    def _register_with_shutdown_manager(self) -> None:
        """Register with the graceful shutdown manager."""
        try:
            from backend.core.resilience.graceful_shutdown import get_shutdown_manager

            manager = get_shutdown_manager()
            if manager:
                manager.register_callback(
                    name="embedding_service_cleanup",
                    callback=self._async_cleanup,
                    priority=30,  # Clean up before database connections
                )
                logger.debug("[EmbeddingService] Registered with GracefulShutdownManager")
        except ImportError:
            logger.debug("[EmbeddingService] GracefulShutdownManager not available")
        except Exception as e:
            logger.debug(f"[EmbeddingService] Could not register with shutdown manager: {e}")

    def _register_with_cross_repo_cleanup(self) -> None:
        """Register with the cross-repo cleanup coordinator."""
        try:
            from backend.core.cross_repo_cleanup import (
                register_cleanup_callback,
                register_resource,
            )
            
            # Register this service for cleanup
            register_cleanup_callback(
                "embedding_service",
                self._sync_cleanup,
            )
            logger.debug("[EmbeddingService] Registered with CrossRepoCleanupCoordinator")
        except ImportError:
            logger.debug("[EmbeddingService] CrossRepoCleanupCoordinator not available")
        except Exception as e:
            logger.debug(f"[EmbeddingService] Could not register with cross-repo cleanup: {e}")

    async def _check_memory_budget(
        self,
        *,
        component: str = "sentence_transformer",
        estimated_mb: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Gate a tier load through the ProactiveResourceGuard.

        Slice 259 — parameterised so each tier asks for its own footprint
        (HIGH ~800MB / LITE ~200MB) and so the background promotion poller can
        pass a short ``timeout`` (no 30s boot-style wait off the hot path).
        Returns True (proceed) if the guard is unavailable — the guard is an
        optimisation, not a hard dependency.
        """
        try:
            from backend.core.proactive_resource_guard import (
                get_proactive_resource_guard,
                COMPONENT_MEMORY_ESTIMATES,
            )

            guard = get_proactive_resource_guard()
            mb = estimated_mb if estimated_mb is not None else \
                COMPONENT_MEMORY_ESTIMATES.get(component, 800)

            kwargs: Dict[str, Any] = dict(
                component=component,
                estimated_mb=mb,
                priority=60,  # Medium-high (embeddings matter)
                can_unload=True,
                unload_callback=self._sync_cleanup,
            )
            if timeout is not None:
                kwargs["timeout"] = timeout
            granted = await guard.request_memory_budget(**kwargs)

            if not granted:
                logger.warning(
                    "[EmbeddingService] Memory budget denied for %s (need ~%dMB)",
                    component, mb,
                )
            return granted

        except ImportError:
            logger.debug("[EmbeddingService] ProactiveResourceGuard not available, skipping memory check")
            return True  # Proceed without guard
        except Exception as e:
            logger.warning(f"[EmbeddingService] Memory check failed: {e}, proceeding anyway")
            return True

    def _make_fastembed_model(self, factory: Any = None) -> "_FastembedSTAdapter":
        """Slice 153 — build the SentenceTransformer.encode-compatible fastembed
        adapter (config-sourced model name). ``factory`` injectable for tests."""
        return _FastembedSTAdapter(self._config.fastembed_model_name, factory=factory)

    # ── Tier observability ──────────────────────────────────────────────
    @property
    def active_tier(self) -> EmbeddingTier:
        """The embedding tier currently serving encode() (NONE if unloaded)."""
        return self._active_tier

    @property
    def tier_name(self) -> str:
        return self._active_tier.name

    def tier_status(self) -> Dict[str, Any]:
        """Snapshot for /observability — adaptive tiering state."""
        return {
            "active_tier": self._active_tier.name,
            "model_loaded": self._model is not None,
            "adaptive_tiering_enabled": self._config.adaptive_tiering_enabled,
            "promotion_enabled": self._config.promotion_enabled,
            "promotion_loop_running": (
                self._promotion_task is not None and not self._promotion_task.done()
            ),
            "promotion_stable_count": self._promotion_stable_count,
            "tier_transitions": self._tier_transitions,
            "promotion_unavailable_reason": self._promotion_unavailable_reason,
        }

    # ── Promotion breaker ─────────────────────────────────────────────
    @staticmethod
    def _import_fault(exc: BaseException) -> Optional[str]:
        """The ImportError in ``exc``'s causal chain, if any, as text.
        Walks ``__cause__``/``__context__`` because a constructor may wrap
        the import failure before it reaches the promotion seam."""
        seen: set = set()
        cur: Optional[BaseException] = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, ImportError):
                return f"{type(cur).__name__}: {cur}"
            cur = cur.__cause__ or cur.__context__
        return None

    def _disable_promotion(self, reason: str) -> None:
        """Trip the breaker: no further LITE→HIGH attempts this process.
        Logged ONCE at WARNING with what would re-enable it."""
        if self._promotion_unavailable_reason is not None:
            return
        self._promotion_unavailable_reason = reason
        logger.warning(
            "[EmbeddingService] HIGH tier unavailable (%s) — LITE→HIGH "
            "promotion disabled for the life of this process; install the "
            "package and restart to re-enable.", reason,
        )

    # ── Model-construction seams (overridable for tests) ────────────────
    def _load_sentence_transformer(self) -> Any:
        """Construct the HIGH-tier PyTorch model. Isolated so tests can patch
        it without a torch install. Raises ImportError if torch is absent."""
        from sentence_transformers import SentenceTransformer  # deferred
        return SentenceTransformer(self._config.model_name, device=self._config.device)

    # ── Memory headroom probes (psutil via the guard, with fallback) ────
    def _available_gb(self) -> Optional[float]:
        """Best-effort system available memory in GB. Reuses the guard's
        accounting (same psutil source) so the tiering and the guard never
        disagree; falls back to psutil, then None (unknown)."""
        try:
            from backend.core.proactive_resource_guard import (
                get_proactive_resource_guard,
            )
            _, available_gb, _ = get_proactive_resource_guard().get_memory_info()
            return float(available_gb)
        except Exception:
            try:
                import psutil
                return psutil.virtual_memory().available / (1024 ** 3)
            except Exception:
                return None

    def _pytorch_headroom_available(self) -> bool:
        """True iff the host can hold the HIGH model AND keep the promotion
        floor afterwards. Conservative: unknown memory → not enough."""
        avail = self._available_gb()
        if avail is None:
            return False
        need = (self._config.pytorch_estimate_mb / 1024.0) + self._config.promotion_headroom_gb
        return avail >= need

    def _lite_headroom_available(self) -> bool:
        """True iff there's room for the LITE model + a small floor. Lenient:
        unknown memory → allow (LITE is the whole point of degrading)."""
        avail = self._available_gb()
        if avail is None:
            return True
        need = (self._config.fastembed_estimate_mb / 1024.0) + self._config.lite_floor_gb
        return avail >= need

    async def _construct_model_offloaded(self, constructor, label: str) -> Any:
        """Slice 26 (2026-07-15, closes the bt-2026-07-15-192117 on-loop model
        load): run a blocking model constructor in the default executor so the
        ~800MB HIGH-tier torch load (and the LITE ONNX construct) never executes
        on the asyncio loop thread. The load was the LAST on-loop heavyweight in
        this service — ``encode()`` was already executor-offloaded. C-extension
        weight loading releases the GIL for its native/IO phases, so a thread
        hop frees the loop the same way the encode path does.

        Kill switch ``JARVIS_EMBED_MODEL_LOAD_OFFLOAD_ENABLED=0`` restores the
        legacy inline construct. Exceptions propagate unchanged — every caller
        already owns tier-demotion / budget-release handling.
        """
        raw = os.environ.get("JARVIS_EMBED_MODEL_LOAD_OFFLOAD_ENABLED", "true")
        if raw.strip().lower() in ("0", "false", "no", "off"):
            return constructor()
        logger.debug("[EmbeddingService] constructing %s tier model off-loop", label)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, constructor)

    async def _load_model(self) -> bool:
        """Lazy-load an embedding tier, adapting to host memory.

        Order: HIGH (PyTorch ~800MB) → on denial/absence, demote to LITE
        (fastembed ~200MB, ONNX/CoreML) rather than running without embeddings.
        Once on LITE, a background poller tries to climb back to HIGH when the
        host recovers headroom. Returns True if any tier is serving.
        """
        if self._model is not None:
            return True
        if self._shutdown_requested:
            logger.warning("[EmbeddingService] Cannot load model during shutdown")
            return False

        # Tier 0 (HIGH): PyTorch SentenceTransformer.
        if await self._try_load_pytorch_tier():
            return True

        # Adaptive demotion — pivot to the lighter ONNX/CoreML tier instead of
        # degrading to no embeddings.
        if self._config.adaptive_tiering_enabled:
            if await self._try_load_fastembed_tier(reason="high_tier_unavailable"):
                self._ensure_promotion_loop()
                return True

        logger.warning(
            "[EmbeddingService] No embedding tier could load — long-term memory "
            "runs without semantic embeddings until memory pressure eases."
        )
        return False

    async def _try_load_pytorch_tier(self) -> bool:
        """Load the HIGH (PyTorch) tier. Returns False (without erroring) when
        the budget is denied or torch is absent, so the caller can demote."""
        # Memory Control Plane: prefer the broker when present.
        try:
            from backend.core.memory_budget_broker import get_memory_budget_broker
            _broker = get_memory_budget_broker()
            if _broker is not None:
                ok = await self._load_model_via_broker(_broker)
                if ok:
                    self._active_tier = EmbeddingTier.HIGH
                return ok  # broker denial → False → caller demotes
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[EmbeddingService] Broker unavailable, using legacy path: {e}")

        # Legacy guard gate.
        if not await self._check_memory_budget(
            component="sentence_transformer",
            estimated_mb=self._config.pytorch_estimate_mb,
        ):
            logger.warning(
                "[EmbeddingService] HIGH tier (PyTorch ~%dMB) denied under memory "
                "pressure — attempting LITE tier.",
                self._config.pytorch_estimate_mb,
            )
            return False

        async with self._model_lock:
            if self._model is not None:
                return True
            try:
                logger.info(f"[EmbeddingService] Loading HIGH tier: {self._config.model_name}")
                start = time.time()
                self._model = await self._construct_model_offloaded(
                    self._load_sentence_transformer, "HIGH",
                )
                self._active_tier = EmbeddingTier.HIGH
                logger.info(
                    "[EmbeddingService] ✅ HIGH tier loaded in %.2fs (device: %s)",
                    time.time() - start, self._config.device,
                )
                return True
            except ImportError as e:
                # torch absent — release the reservation we took and demote.
                logger.warning(
                    "[EmbeddingService] sentence-transformers unavailable (%s) — "
                    "demoting to fastembed LITE tier.", e,
                )
                self._release_component("sentence_transformer")
                # The same absence that will fail every promotion poll: trip
                # the breaker HERE so the poller is never even started.
                self._disable_promotion(f"{type(e).__name__}: {e}")
                return False
            except Exception as e:
                logger.error(f"[EmbeddingService] ❌ HIGH tier load failed: {e}")
                self._release_component("sentence_transformer")
                return False

    async def _try_load_fastembed_tier(self, *, reason: str) -> bool:
        """Load the LITE (fastembed / ONNX-CoreML) tier. Returns False only if
        memory is too tight even for ~200MB or fastembed is unavailable."""
        if not self._lite_headroom_available():
            logger.warning(
                "[EmbeddingService] Even the LITE tier (~%dMB) exceeds free "
                "memory — running without embeddings.",
                self._config.fastembed_estimate_mb,
            )
            return False
        async with self._model_lock:
            if self._model is not None:
                return True
            try:
                start = time.time()
                self._model = await self._construct_model_offloaded(
                    self._make_fastembed_model, "LITE",
                )
                prev = self._active_tier
                self._active_tier = EmbeddingTier.LITE
                if prev != EmbeddingTier.LITE:
                    self._tier_transitions += 1
                logger.warning(
                    "[EmbeddingService] ⬇️ Serving on LITE tier (fastembed=%s, "
                    "~%dMB ONNX/CoreML) reason=%s — loaded in %.2fs. Will promote "
                    "back to HIGH when headroom returns.",
                    self._config.fastembed_model_name,
                    self._config.fastembed_estimate_mb, reason, time.time() - start,
                )
                return True
            except Exception as fe:  # noqa: BLE001 — fastembed missing / model fetch failed
                logger.warning(
                    "[EmbeddingService] LITE tier (fastembed) unavailable: %s", fe,
                )
                return False

    def _release_component(self, component: str) -> None:
        """Best-effort release of a guard budget reservation. NEVER raises."""
        try:
            from backend.core.proactive_resource_guard import (
                get_proactive_resource_guard,
            )
            get_proactive_resource_guard().release_budget(component)
        except Exception:  # noqa: BLE001
            pass

    # ── Heuristic Promotion Protocol (LITE → HIGH) ──────────────────────
    async def maybe_promote_tier(self) -> bool:
        """Attempt a single LITE→HIGH promotion. Returns True if promoted.

        Guards: only from LITE, only when enabled, only after
        ``promotion_stable_checks`` consecutive headroom observations
        (hysteresis), and the HIGH load is gated again at commit time. Atomic:
        the model handle is swapped under the lock so in-flight encode() never
        sees a half-loaded model. NEVER raises.
        """
        if self._shutdown_requested:
            return False
        if self._active_tier != EmbeddingTier.LITE:
            return False
        if not (self._config.adaptive_tiering_enabled and self._config.promotion_enabled):
            return False
        if self._promotion_unavailable_reason is not None:
            return False                     # breaker tripped — nothing to poll

        if not self._pytorch_headroom_available():
            self._promotion_stable_count = 0
            return False
        self._promotion_stable_count += 1
        if self._promotion_stable_count < self._config.promotion_stable_checks:
            return False

        # Gate the actual reservation (short timeout — we already saw headroom).
        if not await self._check_memory_budget(
            component="sentence_transformer",
            estimated_mb=self._config.pytorch_estimate_mb,
            timeout=2.0,
        ):
            self._promotion_stable_count = 0
            return False

        async with self._model_lock:
            if self._active_tier != EmbeddingTier.LITE:  # raced
                self._release_component("sentence_transformer")
                return False
            old_model = self._model
            try:
                start = time.time()
                new_model = await self._construct_model_offloaded(
                    self._load_sentence_transformer, "HIGH(promotion)",
                )
            except Exception as e:  # noqa: BLE001
                self._release_component("sentence_transformer")
                self._promotion_stable_count = 0
                fault = self._import_fault(e)
                if fault is not None:
                    self._disable_promotion(fault)   # permanent: the package is absent
                else:
                    logger.warning(
                        "[EmbeddingService] Promotion HIGH load failed (%s) — staying on LITE.", e,
                    )
                return False
            self._model = new_model
            self._active_tier = EmbeddingTier.HIGH
            self._promotion_stable_count = 0
            self._tier_transitions += 1
            logger.info(
                "[EmbeddingService] ⬆️ Promoted LITE→HIGH (fastembed→PyTorch) in "
                "%.2fs — memory headroom recovered.", time.time() - start,
            )
        # Free the LITE model outside the lock.
        with suppress(Exception):
            del old_model
            gc.collect()
        return True

    def _ensure_promotion_loop(self) -> None:
        """Start the background LITE→HIGH poller once, if enabled and a loop
        is running. Best-effort — promotion is opportunistic, never required."""
        if not (self._config.adaptive_tiering_enabled and self._config.promotion_enabled):
            return
        if self._promotion_unavailable_reason is not None:
            return                           # breaker tripped — no poller
        if self._promotion_task is not None and not self._promotion_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop — caller can retry on next load
        self._promotion_task = loop.create_task(self._promotion_loop())

    async def _promotion_loop(self) -> None:
        """Poll for headroom while degraded; exit once promoted or on shutdown."""
        try:
            while (
                not self._shutdown_requested
                and self._active_tier == EmbeddingTier.LITE
                and self._promotion_unavailable_reason is None
            ):
                await asyncio.sleep(self._config.promotion_poll_s)
                if self._shutdown_requested:
                    break
                try:
                    if await self.maybe_promote_tier():
                        break
                except Exception as e:  # noqa: BLE001
                    logger.debug("[EmbeddingService] promotion attempt error (ignored): %s", e)
        except asyncio.CancelledError:
            pass
        finally:
            self._promotion_task = None

    async def _load_model_via_broker(self, broker) -> bool:
        """Load embedding model via Memory Control Plane broker."""
        from backend.core.budgeted_loaders import EmbeddingBudgetedLoader

        loader = EmbeddingBudgetedLoader()
        estimate = loader.estimate_bytes({})

        try:
            grant = await broker.request(
                component=loader.component_id,
                bytes_requested=estimate,
                priority=loader.priority,
                phase=loader.phase,
            )
        except Exception as e:
            logger.warning(f"[EmbeddingService] Broker denied embedding grant: {e}")
            return False

        async with grant:
            result = await loader.load_with_grant(grant)
            if result.success and result.model_handle is not None:
                self._model = result.model_handle
                await grant.commit(result.actual_bytes, result.config_proof)
                self._active_grant = grant
                logger.info(
                    f"[EmbeddingService] Loaded via broker grant "
                    f"(bytes={grant.granted_bytes})"
                )
                return True
            else:
                logger.error(
                    f"[EmbeddingService] Broker-mediated load failed: {result.error}"
                )
                return False

    async def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        convert_to_numpy: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Encode texts to embeddings.

        Args:
            texts: Text or list of texts to encode
            batch_size: Override batch size for this call
            normalize: Override normalization setting
            convert_to_numpy: Convert to numpy array

        Returns:
            Numpy array of shape (n_texts, embedding_dim) or None on error
        """
        if self._shutdown_requested:
            logger.warning("[EmbeddingService] Cannot encode during shutdown")
            return None

        # Ensure model is loaded
        if not await self._load_model():
            return None

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        if self._config.enable_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = str(hash(text))  # Convert to string for type safety
                if cache_key in self._cache:
                    cached_results.append((i, self._cache[cache_key]))
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1

            # If all cached, return immediately
            if not uncached_texts:
                result = np.zeros((len(texts), cached_results[0][1].shape[0]))
                for i, emb in cached_results:
                    result[i] = emb
                return result

            texts_to_encode = uncached_texts
        else:
            texts_to_encode = list(texts)
            uncached_indices = list(range(len(texts)))
            cached_results = []

        try:
            # Run encoding in thread pool to avoid blocking event loop
            # Note: self._model is guaranteed to be loaded by _load_model() check above
            model = self._model
            if model is None:
                logger.error("[EmbeddingService] Model not loaded")
                return None

            loop = asyncio.get_running_loop()
            embeddings = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: model.encode(
                        texts_to_encode,
                        batch_size=batch_size or self._config.batch_size,
                        normalize_embeddings=normalize if normalize is not None else self._config.normalize_embeddings,
                        show_progress_bar=self._config.show_progress_bar,
                        convert_to_numpy=convert_to_numpy,
                    )
                ),
                timeout=self._config.encode_timeout,
            )

            self._encode_count += len(texts_to_encode)

            # Update cache
            if self._config.enable_cache:
                for i, text in enumerate(texts_to_encode):
                    cache_key = str(hash(text))  # Convert to string for type safety
                    if len(self._cache) < self._config.cache_maxsize:
                        self._cache[cache_key] = embeddings[i]

            # Merge cached and new results
            if cached_results:
                result = np.zeros((len(texts), embeddings.shape[1]))
                # Fill in cached
                for i, emb in cached_results:
                    result[i] = emb
                # Fill in new
                for i, idx in enumerate(uncached_indices):
                    result[idx] = embeddings[i]
                return result

            return embeddings

        except asyncio.TimeoutError:
            logger.error(f"[EmbeddingService] Encoding timed out after {self._config.encode_timeout}s")
            return None
        except Exception as e:
            logger.error(f"[EmbeddingService] Encoding failed: {e}")
            return None

    def encode_sync(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Synchronous encoding (for non-async contexts).

        Prefer encode() when possible.
        """
        with self._thread_lock:
            if self._model is None:
                # Try to load via async path if possible
                try:
                    asyncio.get_running_loop()
                    # We're in an async context - can't load synchronously
                    logger.warning(
                        "[EmbeddingService] encode_sync called without model loaded. "
                        "Call await _load_model() first."
                    )
                    return None
                except RuntimeError:
                    # No event loop running - load synchronously (legacy fallback)
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.warning(
                            "[EmbeddingService] Loading SentenceTransformer synchronously "
                            "(legacy fallback - prefer async _load_model())"
                        )
                        self._model = SentenceTransformer(
                            self._config.model_name,
                            device=self._config.device,
                        )
                    except Exception as e:
                        logger.error(f"[EmbeddingService] Sync model load failed: {e}")
                        return None

            if isinstance(texts, str):
                texts = [texts]

            try:
                return self._model.encode(
                    texts,
                    batch_size=batch_size or self._config.batch_size,
                    normalize_embeddings=normalize if normalize is not None else self._config.normalize_embeddings,
                    show_progress_bar=False,
                )
            except Exception as e:
                logger.error(f"[EmbeddingService] Sync encoding failed: {e}")
                return None

    async def _async_cleanup(self) -> None:
        """
        Async cleanup called by GracefulShutdownManager.

        CRITICAL: This properly cleans up SentenceTransformer resources to prevent
        semaphore leaks.
        """
        self._shutdown_requested = True
        logger.info("[EmbeddingService] Starting cleanup...")

        # Slice 259 — stop the background tier-promotion poller first.
        _ptask = self._promotion_task
        if _ptask is not None and not _ptask.done():
            _ptask.cancel()
            with suppress(Exception):
                await _ptask
        self._promotion_task = None
        self._active_tier = EmbeddingTier.NONE

        try:
            # Stop any multiprocess pools that may have been started
            if self._model is not None:
                # SentenceTransformer's stop_multi_process_pool() if available
                if hasattr(self._model, 'stop_multi_process_pool'):
                    try:
                        self._model.stop_multi_process_pool()
                        logger.debug("[EmbeddingService] Stopped multiprocess pool")
                    except Exception as e:
                        logger.debug(f"[EmbeddingService] Pool stop error (may be fine): {e}")

                # Clear model reference to allow garbage collection
                self._model = None

            # Clear cache
            self._cache.clear()

            # Force garbage collection to clean up any remaining references
            gc.collect()

            logger.info(
                f"[EmbeddingService] ✅ Cleanup complete "
                f"(encoded {self._encode_count} texts, "
                f"cache hits: {self._cache_hits}, misses: {self._cache_misses})"
            )

        except Exception as e:
            logger.error(f"[EmbeddingService] Cleanup error: {e}")

    def _sync_cleanup(self) -> None:
        """
        Synchronous cleanup for atexit.

        Called when Python interpreter is shutting down.
        """
        self._shutdown_requested = True

        # Slice 259 — signal the promotion poller to exit (it checks
        # _shutdown_requested each tick); can't await from a sync context.
        with suppress(Exception):
            if self._promotion_task is not None and not self._promotion_task.done():
                self._promotion_task.cancel()
        self._active_tier = EmbeddingTier.NONE

        try:
            if self._model is not None:
                if hasattr(self._model, 'stop_multi_process_pool'):
                    with suppress(Exception):
                        self._model.stop_multi_process_pool()
                self._model = None

            self._cache.clear()
            gc.collect()

            logger.debug("[EmbeddingService] atexit cleanup complete")
        except Exception:
            pass  # Swallow errors during interpreter shutdown

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "model_loaded": self._model is not None,
            "model_name": self._config.model_name,
            "device": self._config.device,
            "active_tier": self._active_tier.name,
            "tier_transitions": self._tier_transitions,
            "encode_count": self._encode_count,
            "cache_enabled": self._config.enable_cache,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "shutdown_requested": self._shutdown_requested,
        }

    @classmethod
    def get_instance(cls) -> Optional["EmbeddingService"]:
        """Get the singleton instance if it exists."""
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._sync_cleanup()
                cls._instance._initialized = False
            cls._instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_service_instance: Optional[EmbeddingService] = None
_service_lock = asyncio.Lock()


async def get_embedding_service(
    config: Optional[EmbeddingServiceConfig] = None,
) -> EmbeddingService:
    """
    Get the centralized embedding service.

    This is the PREFERRED way to access the embedding service.
    The service is lazily initialized on first call.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton EmbeddingService instance
    """
    global _service_instance

    if _service_instance is None:
        async with _service_lock:
            if _service_instance is None:
                _service_instance = EmbeddingService(config)

    return _service_instance


async def encode_texts(
    texts: Union[str, Sequence[str]],
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Convenience function to encode texts using the shared service.

    Args:
        texts: Text or list of texts to encode
        **kwargs: Additional arguments passed to encode()

    Returns:
        Numpy array of embeddings or None on error
    """
    service = await get_embedding_service()
    return await service.encode(texts, **kwargs)


def encode_texts_sync(
    texts: Union[str, Sequence[str]],
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Synchronous convenience function for encoding.

    Use encode_texts() when possible.
    """
    service = EmbeddingService()
    return service.encode_sync(texts, **kwargs)


async def cleanup_embedding_service() -> None:
    """
    Explicitly cleanup the embedding service.

    Called automatically during shutdown, but can be called manually if needed.
    """
    global _service_instance

    if _service_instance is not None:
        await _service_instance._async_cleanup()
        _service_instance = None


# =============================================================================
# MULTIPROCESSING SEMAPHORE CLEANUP
# =============================================================================

def cleanup_torch_multiprocessing() -> int:
    """
    Clean up any orphaned torch.multiprocessing resources.

    This is a last-resort cleanup that should be called during shutdown
    to prevent semaphore leak warnings.

    Returns:
        Number of resources cleaned up
    """
    cleaned = 0

    try:
        import torch.multiprocessing as mp

        # Check for any active pools and terminate them
        # This is aggressive but necessary to prevent leaks
        if hasattr(mp, '_children'):
            for child in list(getattr(mp, '_children', {}).values()):
                try:
                    if hasattr(child, 'terminate'):
                        child.terminate()
                        cleaned += 1
                except Exception:
                    pass

        # Force garbage collection to clean up semaphores
        gc.collect()

    except ImportError:
        pass  # torch not installed
    except Exception as e:
        logger.debug(f"[EmbeddingService] torch.multiprocessing cleanup error: {e}")

    return cleaned


# =============================================================================
# MODULE-LEVEL REGISTRATION
# =============================================================================

def _register_cleanup_handlers() -> None:
    """Register cleanup handlers at module load."""

    def cleanup_on_exit():
        """Cleanup handler for atexit."""
        cleanup_torch_multiprocessing()
        if _service_instance:
            _service_instance._sync_cleanup()

    # Guarded: KeyboardInterrupt/SystemExit are BaseExceptions, so the
    # `except Exception` inside these handlers never caught them and a
    # Ctrl+C landing here printed a traceback over the goodbye. Imported
    # locally so this can never introduce an import cycle.
    from backend.core.ouroboros.governance.exit_guard import (
        guarded_atexit_register,
    )
    guarded_atexit_register(cleanup_on_exit)
# Auto-register cleanup handlers when module is imported
_register_cleanup_handlers()
