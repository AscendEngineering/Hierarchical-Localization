"""
General-purpose LRU cache with GPU memory management.

Provides a data-agnostic caching mechanism that handles GPU tensor cleanup
on eviction. Works with any data type: tensors, dicts, numpy arrays, etc.
"""

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, TypeVar

import torch

T = TypeVar("T")


class LRUCache:
    """
    General-purpose LRU cache with automatic GPU memory cleanup.
    
    Works with any data type. Automatically moves tensors to device and
    frees GPU memory when items are evicted.
    """
    
    def __init__(
        self,
        max_items: int = 200,
        device: str = "cuda",
        move_to_device: bool = True,
    ):
        """
        Args:
            max_items: Maximum number of items to keep in cache
            device: Target device for tensors ("cuda", "cpu", etc.)
            move_to_device: Whether to automatically move tensors to device
        """
        # Store configuration
        self.max_items = max_items
        self.device = device
        self.move_to_device = move_to_device
        
        # Initialize cache storage (OrderedDict maintains insertion order)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        
        # Initialize statistics counters
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, load_fn: Optional[Callable[[str], T]] = None) -> Optional[T]:
        """
        Get item from cache.
        
        If key exists, returns cached value and marks as recently used.
        If key doesn't exist and load_fn is provided, loads data, caches it, and returns.
        If key doesn't exist and no load_fn, returns None.
        
        Args:
            key: Cache key
            load_fn: Optional function to load data on cache miss.
                     Signature: (key: str) -> data
        
        Returns:
            Cached or newly loaded data, or None if not found and no load_fn
        """
        # If key exists, cache hit
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        
        # If load data is not provided, return None
        if load_fn is None:
            return None
        
        # Cache miss: load data, cache it, and return
        self._misses += 1
        data = load_fn(key)
        self.put(key, data)
        return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Put item into cache.
        
        Automatically moves tensors to device (if enabled) and evicts
        oldest items if cache is full.
        
        Args:
            key: Cache key
            value: Data to cache (any type)
        """
        # Move to device if needed
        if self.move_to_device:
            value = self._to_device(value)
        
        # If key exists, remove it first (will be re-added at end)
        if key in self._cache:
            del self._cache[key]
        
        # Evict oldest items if full
        while len(self._cache) >= self.max_items:
            self._evict_oldest()
        
        # Store the value in cache
        self._cache[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
    
    def _to_device(self, data: Any) -> Any:
        """Recursively move tensors to device."""
        # If tensor, move directly to device
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        
        # If dict, recursively process each value
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        
        # If list/tuple, recursively process and preserve type
        elif isinstance(data, (list, tuple)):
            result = [self._to_device(v) for v in data]
            return type(data)(result)
        
        # For other types (int, str, etc.), return as-is
        return data
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) item."""
        # If cache is empty, nothing to evict
        if not self._cache:
            return
        
        # Pop oldest item (first in OrderedDict) and free its memory
        _, evicted = self._cache.popitem(last=False)
        self._free_memory(evicted)
    
    def _free_memory(self, data: Any) -> None:
        """Recursively delete tensor data to free GPU memory."""
        # If tensor, delete to release GPU memory
        if isinstance(data, torch.Tensor):
            del data
        
        # If dict, recursively free each value
        elif isinstance(data, dict):
            for v in data.values():
                self._free_memory(v)
        
        # If list/tuple, recursively free each element
        elif isinstance(data, (list, tuple)):
            for v in data:
                self._free_memory(v)
    
    def clear(self) -> None:
        """Clear all items from cache and free GPU memory."""
        # Free memory for all cached items
        for value in self._cache.values():
            self._free_memory(value)
        
        # Clear the cache storage
        self._cache.clear()
        
        # Reset statistics
        self._hits = 0
        self._misses = 0
        
        # If using CUDA, explicitly empty the GPU cache
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        # Calculate hit rate (avoid division by zero)
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        # Return stats dictionary
        return {
            "size": len(self._cache),
            "max_size": self.max_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
        }
    
    def keys(self):
        """Return cache keys (in LRU order, oldest first)."""
        return self._cache.keys()
