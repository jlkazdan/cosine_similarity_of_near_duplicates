import numpy as np
import hashlib
import json
import os
import time
from pathlib import Path

# Try to import multiple translation libraries for fallback
TRANSLATION_APIS = []

# Try googletrans first (most common)
try:
    from googletrans import Translator as GoogleTranslator
    googletrans_translator = GoogleTranslator()
    TRANSLATION_APIS.append(('googletrans', googletrans_translator))
    print("✓ googletrans available")
except Exception as e:
    print(f"✗ googletrans not available: {e}")

# Try deep-translator as backup
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    TRANSLATION_APIS.append(('deep-translator', DeepGoogleTranslator))
    print("✓ deep-translator available")
except Exception as e:
    print(f"✗ deep-translator not available: {e}")

# Try translate library as third option
try:
    from translate import Translator as TranslateTranslator
    TRANSLATION_APIS.append(('translate', TranslateTranslator))
    print("✓ translate library available")
except Exception as e:
    print(f"✗ translate library not available: {e}")

if not TRANSLATION_APIS:
    print("WARNING: No translation APIs available! Install with:")
    print("  pip install googletrans==4.0.0-rc1")
    print("  pip install deep-translator")

# Cache file path - relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / "data" / "transformation_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_text_hash(text: str) -> str:
    """Generate a hash for the input text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cache_path(transformation: str) -> Path:
    """Get the cache file path for a specific transformation."""
    return CACHE_DIR / f"{transformation}_cache.json"

def load_cache(transformation: str) -> dict:
    """Load the cache for a specific transformation."""
    cache_path = get_cache_path(transformation)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(transformation: str, cache: dict):
    """Save the cache for a specific transformation."""
    cache_path = get_cache_path(transformation)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_from_cache_or_compute(text: str, transformation: str, compute_fn):
    """
    Get transformation from cache or compute it.
    RAISES EXCEPTION if transformation fails.
    """
    # Load cache
    cache = load_cache(transformation)
    
    # Generate hash
    text_hash = get_text_hash(text)
    
    # Check if result is in cache
    if text_hash in cache:
        print(f"  [Cache hit] Using cached result for {transformation}")
        return cache[text_hash]
    
    print(f"  [Cache miss] Computing {transformation}...")
    
    # Compute transformation - let exceptions propagate
    result = compute_fn(text)
    
    # Validate result is a string
    if not isinstance(result, str):
        error_msg = f"{transformation} returned non-string type {type(result)}"
        print(f"Error: {error_msg}")
        raise TypeError(error_msg)
    
    # Validate result is different from input for transformations that should change it
    if transformation.startswith('translate_') and result == text:
        error_msg = f"{transformation} returned unchanged text (possible translation failure)"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)
    
    # Save to cache
    cache[text_hash] = result
    save_cache(transformation, cache)
    print(f"  [Cached] Saved result for {transformation}")
    
    return result

def capitalize_humps(text):
    """Alternate uppercase/lowercase for each letter."""
    new_text = ""
    for i in range(len(text)):
        if not text[i].isalpha():
            new_text += text[i]
        else:
            if i % 2 == 0:
                new_text += text[i]
            else:
                if text[i].isupper():
                    new_text += chr(ord("a") - ord("A") + ord(text[i]))
                else:
                    new_text += chr(ord("A") - ord("a") + ord(text[i]))
    return new_text

def translate_with_googletrans(text, language, translator, max_retries=3, initial_delay=1.0):
    """Try translation with googletrans library."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(delay)
            else:
                time.sleep(0.3)  # Small delay
            
            translation = translator.translate(text, dest=language)
            if translation and hasattr(translation, 'text') and translation.text:
                return translation.text
            delay *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay *= 2
    return None

def translate_with_deep_translator(text, language, translator_class, max_retries=3):
    """Try translation with deep-translator library."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(1.0 * (attempt + 1))
            
            translator = translator_class(source='auto', target=language)
            result = translator.translate(text)
            if result and result != text:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
    return None

def translate_with_translate_lib(text, language, translator_class, max_retries=3):
    """Try translation with translate library."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(1.0 * (attempt + 1))
            
            translator = translator_class(to_lang=language)
            result = translator.translate(text)
            if result and result != text:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
    return None

def translate(text, language='fr', max_retries_per_api=3):
    """
    Translate text with multiple API fallbacks.
    Tries each available API in order until one succeeds.
    
    Args:
        text: Text to translate
        language: Target language code
        max_retries_per_api: Max retries per API before trying next one
        
    Raises:
        RuntimeError: If all APIs fail
    """
    if not TRANSLATION_APIS:
        raise RuntimeError("No translation APIs available. Please install googletrans or deep-translator.")
    
    # Map language codes
    lang_map = {
        'zh-CN': 'zh-CN',
        'fr': 'fr',
        'de': 'de'
    }
    target_lang = lang_map.get(language, language)
    
    errors = []
    
    for api_name, api_obj in TRANSLATION_APIS:
        try:
            print(f"    Trying {api_name}...")
            
            if api_name == 'googletrans':
                result = translate_with_googletrans(text, target_lang, api_obj, max_retries_per_api)
                if result:
                    print(f"    ✓ Translation successful with {api_name}")
                    return result
                    
            elif api_name == 'deep-translator':
                result = translate_with_deep_translator(text, target_lang, api_obj, max_retries_per_api)
                if result:
                    print(f"    ✓ Translation successful with {api_name}")
                    return result
                    
            elif api_name == 'translate':
                result = translate_with_translate_lib(text, target_lang, api_obj, max_retries_per_api)
                if result:
                    print(f"    ✓ Translation successful with {api_name}")
                    return result
            
            errors.append(f"{api_name}: returned None")
            
        except Exception as e:
            error_msg = f"{api_name}: {type(e).__name__}: {str(e)[:100]}"
            print(f"    ✗ {error_msg}")
            errors.append(error_msg)
            continue
    
    # If we get here, all APIs failed
    error_summary = "; ".join(errors)
    raise RuntimeError(f"All translation APIs failed for {language}. Errors: {error_summary}")

def drop_words(text, prob=0.1):
    """Randomly drop words with given probability."""
    rng = np.random.default_rng()
    arr_text = text.split()
    new_text = ""
    for word in arr_text:
        rand = rng.binomial(n=1, p=1-prob, size=1)
        if rand == 1:
            new_text += word + " "
    return new_text.strip()

def swap_characters(text, prob=0.03):
    """Swap each character with a random character with given probability (simulates typos)."""
    rng = np.random.default_rng()
    # Characters that could be typos - letters, numbers, common punctuation
    possible_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\'"'
    
    new_text = ""
    for char in text:
        rand = rng.binomial(n=1, p=prob, size=1)
        if rand == 1 and char in possible_chars:
            # Swap with a random character
            new_text += rng.choice(list(possible_chars))
        else:
            new_text += char
    return new_text

def text_transformations(text: str, transformation: str):
    """
    Apply text transformation with caching.
    
    Args:
        text: Input text to transform
        transformation: Type of transformation to apply
        
    Returns:
        Transformed text
    """
    if transformation == "none":
        return text

    elif transformation == "capitalize_humps":
        return get_from_cache_or_compute(
            text, 
            transformation, 
            capitalize_humps
        )
    
    elif transformation == "translate_to_chinese":
        return get_from_cache_or_compute(
            text,
            transformation,
            lambda t: translate(t, "zh-CN")
        )
    
    elif transformation == "translate_to_french":
        return get_from_cache_or_compute(
            text,
            transformation,
            lambda t: translate(t, "fr")
        )
    
    elif transformation == "translate_to_german":
        return get_from_cache_or_compute(
            text,
            transformation,
            lambda t: translate(t, "de")
        )

    elif transformation == "drop_words":
        return get_from_cache_or_compute(
            text,
            transformation,
            drop_words
        )
    
    elif transformation == "swap_characters":
        return get_from_cache_or_compute(
            text,
            transformation,
            swap_characters
        )
    
    else:
        raise NotImplementedError(f"Transformation '{transformation}' not implemented")

def clear_cache(transformation: str = None):
    """
    Clear cache for a specific transformation or all transformations.
    
    Args:
        transformation: Specific transformation to clear, or None to clear all
    """
    if transformation:
        cache_path = get_cache_path(transformation)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache for {transformation}")
    else:
        for cache_file in CACHE_DIR.glob("*_cache.json"):
            cache_file.unlink()
        print("Cleared all transformation caches")

def get_cache_stats():
    """Print statistics about the transformation caches."""
    transformations = [
        "capitalize_humps",
        "translate_to_chinese", 
        "translate_to_french",
        "translate_to_german",
        "drop_words",
        "swap_characters"
    ]
    
    print("Transformation Cache Statistics:")
    print("-" * 50)
    for trans in transformations:
        cache = load_cache(trans)
        print(f"{trans}: {len(cache)} cached entries")
    print("-" * 50)

if __name__ == "__main__":
    # Example usage
    test_text = "Hello, this is a test sentence for caching."
    
    print("Testing transformations with caching...")
    print(f"Original: {test_text}\n")
    
    transformations = [
        "capitalize_humps",
        "translate_to_french",
        "translate_to_german",
        "drop_words",
        "swap_characters"
    ]
    
    for trans in transformations:
        result = text_transformations(test_text, trans)
        print(f"{trans}: {result}\n")
    
    print("\n")
    get_cache_stats()