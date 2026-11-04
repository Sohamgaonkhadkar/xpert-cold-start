"""Fast, zero-dependency Devanagari (Hindi) and Bengali Unicode script normalizer.

Python's `re` module treats Indic combining vowel matras (e.g. 'ा' U+093E, 'ी' U+0940, 'া' U+09BE, 'ে' U+09C7)
as non-word characters, causing `\\b` word-boundary regexes to fail silently on native Hindi/Bengali text,
and preventing cross-script lexical matching when a Hindi/Bengali message is answered in English.

`normalize_multilingual_text` is a zero-ASCII-overhead fast path (`text.isascii()` returns immediately)
that converts Indic digits/punctuation and appends canonical Romanized equivalents for Hindi/Bengali
kinship, temporal, commitment, request, and closure phrases alongside the original Unicode text.
"""

from __future__ import annotations

from typing import List, Tuple

# Digit & punctuation translation table (Devanagari ০-९ U+0966..U+096F, Bengali ০-৯ U+09E6..U+09EF, danda ।/॥)
_DIGIT_PUNCT_TRANS = str.maketrans(
    {
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
        "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
        "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
        "।": ".", "॥": ".",
    }
)

# High-value native Devanagari (Hindi) & Bengali phrase/lexicon equivalences -> Romanized tokens
# Sorted longest-first at import time so multi-word idioms match cleanly before single words.
_RAW_PHRASE_PAIRS: List[Tuple[str, str]] = [
    # --- Hindi (Devanagari) closures, cancellations, deferrals & commitments ---
    ("अभी तक नहीं", "abhi tak nahi"),
    ("अभी नहीं हुआ", "abhi nahi hua"),
    ("नहीं हुआ", "nahi hua"),
    ("नहीं भेजा", "nahi bheja"),
    ("नहीं किया", "nahi kiya"),
    ("बाकी है", "baaki hai"),
    ("हो गया क्या", "ho gaya kya"),
    ("भेज दिया क्या", "bhej diya kya"),
    ("कर दिया क्या", "kar diya kya"),
    ("जमा कर दिया", "jama kar diya"),
    ("पे कर दिया", "pay kar diya"),
    ("बुक कर दिया", "book kar diya"),
    ("भेज दिया", "bhej diya"),
    ("कर दिया", "kar diya"),
    ("दे दिया", "de diya"),
    ("ले लिया", "le liya"),
    ("हो गया", "ho gaya"),
    ("मिल गया", "mil gaya"),
    ("मिल गयी", "mil gayi"),
    ("मिल गई", "mil gayi"),
    ("रहने दो", "rehne do"),
    ("मत भेजो", "mat bhejo"),
    ("मत करो", "mat karo"),
    ("ज़रूरत नहीं", "zaroorat nahi"),
    ("जरूरत नहीं", "jarurat nahi"),
    ("कैंसिल कर दो", "cancel kar do"),
    ("मैं देख लूंगा", "main dekh lunga"),
    ("मैं कर लूंगा", "main kar lunga"),
    ("मैं कर लूँगा", "main kar lunga"),
    ("तुम देख लो", "tum dekh lo"),
    ("आप देख लो", "aap dekh lo"),
    ("भेज दूंगा", "bhej dunga"),
    ("भेज दूँगा", "bhej dunga"),
    ("भेज दूंगी", "bhej dungi"),
    ("कर दूंगा", "kar dunga"),
    ("कर दूँगा", "kar dunga"),
    ("कर दूंगी", "kar dungi"),
    ("दे दूंगा", "de dunga"),
    ("ले आऊंगा", "le aaunga"),
    ("भर देना", "pay kar dena"),
    ("भेज देना", "bhej dena"),
    ("कर देना", "kar dena"),
    ("ले आना", "le aana"),
    ("खोज के बताओ", "khoj ke batao"),
    ("पता करके बताओ", "pata karke batao"),
    ("सोचना पड़ेगा", "sochna padega"),
    ("देखना पड़ेगा", "dekhna padega"),
    ("करना पड़ेगा", "karna padega"),
    ("देखता हूँ", "dekhta hoon"),
    ("देखता हूं", "dekhta hoon"),
    ("ठीक है", "theek hai"),
    ("जी साहब", "ji sahab"),
    ("जी भैया", "ji bhaiya"),
    # --- Hindi months ---
    ("जनवरी", "january"), ("फरवरी", "february"), ("फ़रवरी", "february"), ("मार्च", "march"),
    ("अप्रैल", "april"), ("मई", "may"), ("जून", "june"), ("जुलाई", "july"),
    ("अगस्त", "august"), ("सितंबर", "september"), ("सितम्बर", "september"),
    ("अक्टूबर", "october"), ("नवंबर", "november"), ("नवम्बर", "november"),
    ("दिसंबर", "december"), ("दिसम्बर", "december"),
    # --- Hindi temporal & kinship/domain vocabulary ---
    ("अगले हफ्ते", "agle hafte"),
    ("अगले महीने", "agle mahine"),
    ("शाम तक", "shaam tak"),
    ("आज शाम", "aaj shaam"),
    ("आज रात", "aaj raat"),
    ("कल सुबह", "kal subah"),
    ("परसों", "parso"),
    ("कल", "kal"),
    ("आज", "aaj"),
    ("सोमवार", "somwar"), ("मंगलवार", "mangalwar"), ("बुधवार", "budhwar"),
    ("गुरुवार", "guruwar"), ("शुक्रवार", "shukrawar"), ("शनिवार", "shaniwar"), ("रविवार", "raviwar"),
    ("ससुरजी", "sasurji"), ("सासू माँ", "sasu maa"), ("मम्मी", "mummy"), ("माँ", "maa"),
    ("पापा", "papa"), ("बाबा", "baba"), ("भैया", "bhaiya"), ("दीदी", "didi"),
    ("मासी", "masi"), ("चाचा", "chacha"), ("बुआ", "bua"), ("मामा", "mama"),
    ("ड्राइवर", "driver"), ("रसोइया", "cook"), ("माली", "mali"), ("मेड", "maid"),
    ("बिजली", "bijli electricity"), ("बिल", "bill"), ("दवाई", "dawai medicine"),
    ("पगार", "pagar salary"), ("वेतन", "salary"), ("किराया", "rent"), ("फीस", "fees"),
    ("स्कूल", "school"), ("टिकट", "ticket"), ("दुकान", "dokan shop"), ("नंबर", "number"),
    ("हाँ", "haan"), ("हां", "haan"), ("जी", "ji"),
    # --- Bengali (Unicode) closures, cancellations, deferrals & commitments ---
    ("এখনো হয়নি", "ekhono hoyni"),
    ("এখনও হয়নি", "ekhono hoyni"),
    ("হয়ে গেছে কি", "hoye geche ki"),
    ("পাঠিয়েছো কি", "pathiyecho ki"),
    ("পাঠিয়ে দিয়েছি", "pathiye diyechi"),
    ("করে দিয়েছি", "kore diyechi"),
    ("দিয়ে দিয়েছি", "deye diyechi"),
    ("হয়ে গেছে", "hoye geche"),
    ("পেয়ে গেছি", "peye gechi"),
    ("পেয়েছি", "peyechi"),
    ("বাদ দাও", "bad dao"),
    ("আর লাগবে না", "ar lagbe na"),
    ("পাঠাতে হবে না", "pathate hobe na"),
    ("আমি দেখে নিচ্ছি", "ami dekhe nichhi"),
    ("আমি করে নিচ্ছি", "ami kore nichhi"),
    ("তুমি দেখে নাও", "tumi dekhe nao"),
    ("পাঠিয়ে দেব", "pathiye debo"),
    ("পাঠিয়ে দেবো", "pathiye debo"),
    ("করে দেব", "kore debo"),
    ("করে দেবো", "kore debo"),
    ("নিয়ে আসবো", "niye ashbo"),
    ("পাঠিয়ে দিও", "pathiye dio"),
    ("করে দিও", "kore dio"),
    ("এনে দিও", "ene dio"),
    ("জানিও", "janio"),
    ("খোঁজ নিয়ে", "khoj niye"),
    ("ব্যবস্থা করতে হবে", "babostha korte hobe"),
    ("দেখতে হবে", "dekhte hobe"),
    ("পরে বলছি", "pore bolchi"),
    # --- Bengali months ---
    ("জানুয়ারি", "january"), ("ফেব্রুয়ারি", "february"), ("মার্চ", "march"),
    ("এপ্রিল", "april"), ("মে", "may"), ("জুন", "june"), ("জুলাই", "july"),
    ("আগস্ট", "august"), ("সেপ্টেম্বর", "september"), ("অক্টোবর", "october"),
    ("নভেম্বর", "november"), ("ডিসেম্বর", "december"),
    # --- Bengali temporal & kinship/domain vocabulary ---
    ("পরের সপ্তাহ", "porer shoptah"),
    ("পরের মাসে", "porer mashe"),
    ("আগামীকাল", "agamikal"),
    ("আজকের মধ্যে", "ajke modhye"),
    ("বিকেলের মধ্যে", "bikeler modhye"),
    ("পরশু", "porshu"),
    ("কালকে", "kalke"),
    ("কাল", "kal"),
    ("আজ", "aaj"),
    ("সোমবার", "sombar"), ("মঙ্গলবার", "mongolbar"), ("বুধবার", "budhbar"),
    ("বৃহস্পতিবার", "brihospotibar"), ("শুক্রবার", "shukrobar"), ("শনিবার", "shonibar"), ("রবিবার", "robibar"),
    ("শ্বশুর", "sasur"), ("শাশুড়ি", "shashuri"), ("মাসি", "mashi"), ("পিসি", "pishi"),
    ("মেসো", "mesho"), ("কাকা", "kaka"), ("মামা", "mama"), ("দাদা", "dada"), ("দিদি", "didi"),
    ("মা", "maa"), ("বাবা", "baba"), ("ড্রাইভার", "driver"), ("মালি", "mali"), ("দিদিমণি", "didimoni"),
    ("বিদ্যুৎ", "electricity"), ("বিল", "bill"), ("ওষুধ", "oshudh medicine"),
    ("বেতন", "beton salary"), ("স্কুল", "school"), ("মিষ্টি", "misti"), ("দোকান", "dokan"), ("নম্বর", "number"),
    ("হ্যাঁ", "haan"),
    # --- Marathi / Gujarati / Tamil / Telugu / Kannada Unicode kinship, commitments & completion cues ---
    ("पाठवतो", "bhej dunga"), ("पाठवते", "bhej dungi"), ("करून टाकतो", "kar dunga"), ("उद्या", "kal"),
    ("पाठवले", "bhej diya"), ("झाले", "ho gaya"), ("केले", "kar diya"), ("आई", "aai maa"),
    ("मावशी", "mavshi"), ("काका", "kaka"), ("वहिनी", "vahini"), ("ताई", "tai"),
    ("મોકલી આપીશ", "bhej dunga"), ("આવતીકાલે", "kal"),
    ("મોકલી દીધું", "bhej diya"), ("થઈ ગયું", "ho gaya"), ("મમ્મી", "mummy"), ("પપ્પા", "papa"), ("માસી", "masi"),
    ("அனுப்புறேன்", "bhej dunga"), ("அனுப்புகிறேன்", "bhej dunga"), ("நாளைக்கு", "kal"), ("நாளை", "kal"),
    ("அனுப்பிட்டேன்", "bhej diya"), ("முடிந்தது", "ho gaya"), ("அம்மா", "amma"), ("அப்பா", "appa"), ("அண்ணா", "anna"), ("அக்கா", "akka"),
    ("పంపిస్తాను", "bhej dunga"), ("రేపు", "kal"),
    ("పంపాను", "bhej diya"), ("అయిపోయింది", "ho gaya"), ("అమ్మ", "amma"), ("నాన్న", "nanna"), ("అన్న", "anna"), ("అక్క", "akka"),
    ("ಕಳಿಸ್ತೀನಿ", "bhej dunga"), ("ನಾಳೆ", "kal"),
    ("ಕಳಿಸಿದೆ", "bhej diya"), ("ಆಯ್ತು", "ho gaya"), ("ಅಮ್ಮ", "amma"), ("ಅಪ್ಪ", "appa"), ("ಅಣ್ಣ", "anna"), ("ಅಕ್ಕ", "akka"),
]

_SORTED_PHRASES: List[Tuple[str, str]] = sorted(
    _RAW_PHRASE_PAIRS, key=lambda kv: len(kv[0]), reverse=True
)


def has_indic_script(text: str) -> bool:
    if not text or text.isascii():
        return False
    for ch in text:
        if "\u0900" <= ch <= "\u0d7f":
            return True
    return False


def _is_indic_char(ch: str) -> bool:
    return "\u0900" <= ch <= "\u0d7f"


def _contains_bounded(haystack: str, needle: str) -> bool:
    """For short 2-3 char Indic roots (e.g. 'মা', 'মে', 'कल', 'जी'), require non-Indic boundaries
    so 'মা' never matches inside 'মাসি'/'মামা' or 'মে' inside 'মেয়ে'."""
    if len(needle) >= 4:
        return needle in haystack
    start = 0
    nlen = len(needle)
    hlen = len(haystack)
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        left_ok = idx == 0 or not _is_indic_char(haystack[idx - 1])
        right_ok = (idx + nlen >= hlen) or not _is_indic_char(haystack[idx + nlen])
        if left_ok and right_ok:
            return True
        start = idx + 1


_INLINE_MONTH_REPLACEMENTS: List[Tuple[str, str]] = [
    ("जनवरी", "january"), ("फरवरी", "february"), ("फ़रवरी", "february"), ("मार्च", "march"),
    ("अप्रैल", "april"), ("मई", "may"), ("जून", "june"), ("जुलाई", "july"),
    ("अगस्त", "august"), ("सितंबर", "september"), ("सितम्बर", "september"),
    ("अक्टूबर", "october"), ("नवंबर", "november"), ("नवम्बर", "november"),
    ("दिसंबर", "december"), ("दिसम्बर", "december"),
    ("জানুয়ারি", "january"), ("ফেব্রুয়ারি", "february"), ("মার্চ", "march"),
    ("এপ্রিল", "april"), ("জুন", "june"), ("জুলাই", "july"),
    ("আগস্ট", "august"), ("সেপ্টেম্বর", "september"), ("অক্টোবর", "october"),
    ("নভেম্বর", "november"), ("ডিসেম্বর", "december"),
]


def normalize_indic_for_matching(text: str) -> str:
    """Fast-path normalizer: returns ASCII strings untouched; for Devanagari/Bengali text,
    normalizes digits/punctuation, maps Indic month names adjacent to digits ('३ नवंबर' -> '3 november'),
    and appends Romanized phrase equivalents alongside the original Unicode text."""
    if not text or text.isascii():
        return text
    if not has_indic_script(text):
        return text

    norm = text.translate(_DIGIT_PUNCT_TRANS)
    date_inline = norm
    for m_ind, m_eng in _INLINE_MONTH_REPLACEMENTS:
        if m_ind in date_inline:
            date_inline = date_inline.replace(m_ind, m_eng)

    matched_roman: List[str] = []
    if date_inline != norm:
        matched_roman.append(date_inline)
    work = norm
    for native_p, roman_p in _SORTED_PHRASES:
        if _contains_bounded(work, native_p):
            matched_roman.append(roman_p)
            if len(native_p) >= 4:
                work = work.replace(native_p, " ")
    if not matched_roman:
        return norm
    return f"{norm} ({' '.join(matched_roman)})"


def is_pure_indic_bare_ack(text: str) -> bool:
    """Detect bare Hindi/Bengali acknowledgements ('ठीक है', 'हाँ', 'হ্যাঁ', 'জি') that are NOT completions."""
    if not text or text.isascii():
        return False
    stripped = text.translate(_DIGIT_PUNCT_TRANS).strip(" .!?,👍🙏👌")
    return stripped in {
        "हाँ", "हां", "जी", "जी साहब", "जी भैया", "ठीक है", "अच्छा", "ओके",
        "হ্যাঁ", "জি", "ঠিক আছে", "আচ্ছা", "ওকে",
    }
