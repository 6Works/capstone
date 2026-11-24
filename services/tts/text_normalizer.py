"""
Text Normalization and Grapheme-to-Phoneme (G2P) Front-End.
Based on Module 2 ("How Machines Process Text") from the Neural TTS & Voice Cloning Course.

Transforms raw unstructured text containing Non-Standard Words (NSW)
such as numbers, currencies, dates, abbreviations, and technical acronyms
into fluent, spoken phonemic representations for acoustic models.
Includes Context-Aware Homograph Disambiguation (e.g. read/read, lead/lead, live/live).
"""

import re
from typing import List, Dict, Any, Tuple, Optional


class HomographDisambiguator:
    """
    Context-aware homograph disambiguation engine for Text-to-Speech front-ends.
    Based on Chapter 3 ("How Do Speaking AIs Understand Text?") of The Sound of AI course.
    """

    HOMOGRAPHS = {
        "read": {
            "past": {"ipa": "rɛd", "arpabet": "R EH1 D", "rule": "Past participle/past tense context (e.g. 'have read', 'already read')"},
            "present": {"ipa": "riːd", "arpabet": "R IY1 D", "rule": "Present/infinitive tense context (e.g. 'will read', 'to read')"}
        },
        "lead": {
            "metal": {"ipa": "lɛd", "arpabet": "L EH1 D", "rule": "Chemical element / heavy metal (e.g. 'lead pipe', 'lead weight')"},
            "verb": {"ipa": "liːd", "arpabet": "L IY1 D", "rule": "Action to guide or precede (e.g. 'lead the way', 'will lead')"}
        },
        "live": {
            "adj": {"ipa": "laɪv", "arpabet": "L AY1 V", "rule": "Adjective modifying broadcast/stream/performance (e.g. 'live stream', 'live demo')"},
            "verb": {"ipa": "lɪv", "arpabet": "L IH1 V", "rule": "Verb denoting residence or existence (e.g. 'I live in', 'where you live')"}
        },
        "record": {
            "noun": {"ipa": "ˈrɛkərd", "arpabet": "R EH1 K ER0 D", "rule": "Noun denoting document/achievement (e.g. 'world record', 'the record')"},
            "verb": {"ipa": "rɪˈkɔːrd", "arpabet": "R IH0 K AO1 R D", "rule": "Verb denoting capturing audio/data (e.g. 'to record', 'will record')"}
        },
        "present": {
            "noun": {"ipa": "ˈprɛzənt", "arpabet": "P R EH1 Z AH0 N T", "rule": "Noun (gift) or temporal state (e.g. 'birthday present', 'at present')"},
            "verb": {"ipa": "prɪˈzɛnt", "arpabet": "P R IH0 Z EH1 N T", "rule": "Verb meaning to introduce or show (e.g. 'to present', 'present the data')"}
        },
        "wind": {
            "noun": {"ipa": "wɪnd", "arpabet": "W IH1 N D", "rule": "Noun referring to atmospheric air flow (e.g. 'strong wind', 'cold wind')"},
            "verb": {"ipa": "waɪnd", "arpabet": "W AY1 N D", "rule": "Verb meaning to turn, coil or conclude (e.g. 'wind down', 'wind the clock')"}
        }
    }

    @classmethod
    def disambiguate(cls, word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Determines the phonemic pronunciation of a homograph based on syntactic context."""
        w = word.lower()
        if w not in cls.HOMOGRAPHS:
            return None

        prev_w = (prev_word or "").lower()
        next_w = (next_word or "").lower()
        h_data = cls.HOMOGRAPHS[w]

        if w == "read":
            if prev_w in ["have", "has", "had", "already", "been", "was", "were", "recently", "just"]:
                res = dict(h_data["past"])
                res["sense"] = "past"
                return res
            res = dict(h_data["present"])
            res["sense"] = "present"
            return res

        elif w == "live":
            if next_w in ["stream", "streaming", "music", "broadcast", "show", "performance", "feed", "event", "demo", "recording"]:
                res = dict(h_data["adj"])
                res["sense"] = "adjective"
                return res
            if prev_w in ["a", "the", "our", "their"]:
                res = dict(h_data["adj"])
                res["sense"] = "adjective"
                return res
            res = dict(h_data["verb"])
            res["sense"] = "verb"
            return res

        elif w == "lead":
            if next_w in ["pipe", "pipes", "poisoning", "bullet", "bullets", "pencil", "paint", "weight"]:
                res = dict(h_data["metal"])
                res["sense"] = "metal_noun"
                return res
            if prev_w in ["of", "heavy", "toxic"]:
                res = dict(h_data["metal"])
                res["sense"] = "metal_noun"
                return res
            res = dict(h_data["verb"])
            res["sense"] = "verb"
            return res

        elif w == "record":
            if prev_w in ["to", "will", "would", "shall", "can", "could", "please", "must", "and"]:
                res = dict(h_data["verb"])
                res["sense"] = "verb"
                return res
            if prev_w in ["a", "the", "new", "world", "track", "broken", "medical"]:
                res = dict(h_data["noun"])
                res["sense"] = "noun"
                return res
            res = dict(h_data["noun"])
            res["sense"] = "noun"
            return res

        elif w == "present":
            if prev_w in ["to", "will", "would", "shall", "can", "could"]:
                res = dict(h_data["verb"])
                res["sense"] = "verb"
                return res
            res = dict(h_data["noun"])
            res["sense"] = "noun"
            return res

        elif w == "wind":
            if next_w in ["down", "up", "the", "clock"] or prev_w in ["to", "will"]:
                res = dict(h_data["verb"])
                res["sense"] = "verb"
                return res
            res = dict(h_data["noun"])
            res["sense"] = "noun"
            return res

        return None

    @classmethod
    def list_homographs(cls) -> Dict[str, Any]:
        return cls.HOMOGRAPHS


class TextNormalizer:
    """
    Normalizes Non-Standard Words (NSW) into spoken orthography.
    Handles numbers, currencies, units, percentages, abbreviations, and acronyms.
    """

    ONES = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ]
    TENS = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    ]
    ORDINALS = {
        "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
        "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
        "11": "eleventh", "12": "twelfth", "13": "thirteenth", "20": "twentieth",
        "21": "twenty-first", "22": "twenty-second", "23": "twenty-third", "30": "thirtieth"
    }

    ABBREVIATIONS = {
        r"\bDr\.\b": "Doctor",
        r"\bMr\.\b": "Mister",
        r"\bMrs\.\b": "Missus",
        r"\bMs\.\b": "Miz",
        r"\bProf\.\b": "Professor",
        r"\bvs\.\b": "versus",
        r"\be\.g\.,?\b": "for example",
        r"\bi\.e\.,?\b": "that is",
        r"\betc\.\b": "et cetera",
        r"\bdept\.\b": "department",
        r"\bapprox\.\b": "approximately",
        r"\best\.\b": "established",
        r"\bmin\.\b": "minutes",
        r"\bsec\.\b": "seconds",
        r"\bhr\.\b": "hours",
        r"\bhrs\.\b": "hours",
    }

    TECH_ACRONYMS = {
        "API": "A P I",
        "ASR": "A S R",
        "TTS": "T T S",
        "NLP": "N L P",
        "NLU": "N L U",
        "LLM": "L L M",
        "VITS": "V I T S",
        "CPU": "C P U",
        "GPU": "G P U",
        "TPU": "T P U",
        "RAM": "ram",
        "ROM": "rom",
        "SQL": "S Q L",
        "HTTP": "H T T P",
        "HTTPS": "H T T P S",
        "REST": "rest",
        "JSON": "jay son",
        "HTML": "H T M L",
        "CSS": "C S S",
        "JS": "J S",
        "AI": "A I",
        "ML": "M L",
        "WER": "W E R",
        "CER": "C E R",
        "ONNX": "on ix",
        "CUDA": "coo dah",
        "SAPI": "sap ee",
    }

    CURRENCY_SYMBOLS = {
        "$": ("dollar", "dollars"),
        "€": ("euro", "euros"),
        "£": ("pound", "pounds"),
        "₹": ("rupee", "rupees"),
        "¥": ("yen", "yen"),
    }

    UNIT_SYMBOLS = {
        "km/h": "kilometers per hour",
        "mph": "miles per hour",
        "km": "kilometers",
        "m": "meters",
        "cm": "centimeters",
        "mm": "millimeters",
        "kg": "kilograms",
        "g": "grams",
        "mg": "milligrams",
        "ms": "milliseconds",
        "s": "seconds",
        "min": "minutes",
        "h": "hours",
        "gb": "gigabytes",
        "mb": "megabytes",
        "kb": "kilobytes",
        "tb": "terabytes",
        "hz": "hertz",
        "khz": "kilohertz",
        "mhz": "megahertz",
        "ghz": "gigahertz",
    }

    @classmethod
    def num_to_words(cls, n: int) -> str:
        """Converts integer into English spoken words."""
        if n == 0:
            return "zero"
        if n < 0:
            return "negative " + cls.num_to_words(abs(n))

        if n < 20:
            return cls.ONES[n]
        if n < 100:
            rem = cls.ONES[n % 10]
            return cls.TENS[n // 10] + (("-" + rem) if rem else "")
        if n < 1000:
            rem = cls.num_to_words(n % 100)
            return cls.ONES[n // 100] + " hundred" + ((" and " + rem) if rem != "zero" else "")
        if n < 1000000:
            thousands = cls.num_to_words(n // 1000)
            rem = cls.num_to_words(n % 1000)
            return thousands + " thousand" + ((" " + rem) if rem != "zero" else "")
        if n < 1000000000:
            millions = cls.num_to_words(n // 1000000)
            rem = cls.num_to_words(n % 1000000)
            return millions + " million" + ((" " + rem) if rem != "zero" else "")
        billions = cls.num_to_words(n // 1000000000)
        rem = cls.num_to_words(n % 1000000000)
        return billions + " billion" + ((" " + rem) if rem != "zero" else "")

    @classmethod
    def expand_years(cls, year_str: str) -> str:
        """Expands 4-digit years like 2026 -> twenty twenty-six, 1995 -> nineteen ninety-five."""
        try:
            year = int(year_str)
            if 1000 <= year <= 2099:
                century = year // 100
                decade = year % 100
                if decade == 0:
                    return f"{cls.num_to_words(century)} hundred"
                if 2000 <= year <= 2009:
                    return f"two thousand {cls.num_to_words(decade)}"
                return f"{cls.num_to_words(century)} {cls.num_to_words(decade)}"
        except ValueError:
            pass
        return year_str

    @classmethod
    def normalize_currencies(cls, text: str) -> str:
        """Expands currency representations like $4.50 or €100."""
        for symbol, (singular, plural) in cls.CURRENCY_SYMBOLS.items():
            pattern = re.escape(symbol) + r"(\d+(?:\.\d{1,2})?)"
            def _replace_currency(match):
                amount_str = match.group(1)
                if "." in amount_str:
                    parts = amount_str.split(".")
                    main_amt = int(parts[0])
                    cents = int(parts[1].ljust(2, "0")[:2])
                    main_unit = singular if main_amt == 1 else plural
                    cents_word = cls.num_to_words(cents)
                    if main_amt == 0:
                        return f"{cents_word} cents"
                    return f"{cls.num_to_words(main_amt)} {main_unit} and {cents_word} cents"
                else:
                    main_amt = int(amount_str)
                    main_unit = singular if main_amt == 1 else plural
                    return f"{cls.num_to_words(main_amt)} {main_unit}"
            text = re.sub(pattern, _replace_currency, text)
        return text

    @classmethod
    def normalize_percentages(cls, text: str) -> str:
        """Expands 50% or 99.5% into spoken English."""
        def _replace_pct(match):
            val = match.group(1)
            if "." in val:
                parts = val.split(".")
                decimal_str = " ".join([cls.ONES[int(d)] if int(d) < 10 else d for d in parts[1]])
                return f"{cls.num_to_words(int(parts[0]))} point {decimal_str} percent"
            return f"{cls.num_to_words(int(val))} percent"
        return re.sub(r"(\d+(?:\.\d+)?)\s*%", _replace_pct, text)

    @classmethod
    def normalize_units(cls, text: str) -> str:
        """Expands measure units like 10km or 500ms."""
        for unit, expanded in sorted(cls.UNIT_SYMBOLS.items(), key=lambda x: -len(x[0])):
            pattern = r"(\b\d+(?:\.\d+)?)\s*" + re.escape(unit) + r"\b"
            def _replace_unit(m):
                num_str = m.group(1)
                if "." in num_str:
                    words = f"{num_str}"
                else:
                    words = cls.num_to_words(int(num_str))
                return f"{words} {expanded}"
            text = re.sub(pattern, _replace_unit, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def normalize_times(cls, text: str) -> str:
        """Expands times like 10:30am, 4:15 pm, 08:00."""
        pattern = r"\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b"
        def _replace_time(m):
            hour = int(m.group(1))
            minute = int(m.group(2))
            meridiem = m.group(3)
            hour_words = cls.num_to_words(hour)
            if minute == 0:
                min_words = "o'clock" if not meridiem else ""
            elif minute < 10:
                min_words = f"oh {cls.num_to_words(minute)}"
            else:
                min_words = cls.num_to_words(minute)
            res = f"{hour_words} {min_words}".strip()
            if meridiem:
                m_spoken = "A M" if meridiem.upper() == "AM" else "P M"
                res = f"{res} {m_spoken}"
            return res
        return re.sub(pattern, _replace_time, text)

    @classmethod
    def normalize_ordinals(cls, text: str) -> str:
        """Expands 1st, 2nd, 3rd, 4th, 21st, etc."""
        pattern = r"\b(\d+)(st|nd|rd|th)\b"
        def _replace_ord(m):
            num = m.group(1)
            if num in cls.ORDINALS:
                return cls.ORDINALS[num]
            try:
                val = int(num)
                last_two = val % 100
                if 10 <= last_two <= 19:
                    return cls.num_to_words(val) + "th"
                last_digit = val % 10
                base = (val // 10) * 10
                base_str = cls.num_to_words(base) if base > 0 else ""
                ord_str = cls.ORDINALS.get(str(last_digit), "th")
                if base_str:
                    return f"{base_str}-{ord_str}"
                return ord_str
            except Exception:
                return m.group(0)
        return re.sub(pattern, _replace_ord, text, flags=re.IGNORECASE)

    @classmethod
    def normalize_numbers(cls, text: str) -> str:
        """Expands standalone integers and decimal numbers."""
        def _replace_decimal(m):
            parts = m.group(0).split(".")
            integer_part = cls.num_to_words(int(parts[0]))
            digits = " ".join([cls.ONES[int(d)] if int(d) < 10 else d for d in parts[1]])
            return f"{integer_part} point {digits}"

        text = re.sub(r"\b\d+\.\d+\b", _replace_decimal, text)

        def _replace_int(m):
            val_str = m.group(0)
            if len(val_str) == 4 and 1900 <= int(val_str) <= 2099:
                return cls.expand_years(val_str)
            return cls.num_to_words(int(val_str))

        text = re.sub(r"\b\d+\b", _replace_int, text)
        return text

    @classmethod
    def normalize_abbreviations(cls, text: str) -> str:
        """Expands common titles and abbreviations."""
        for pattern, replacement in cls.ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def normalize_acronyms(cls, text: str) -> str:
        """Expands technical acronyms for clear phonetic pronunciation."""
        for acronym, spoken in cls.TECH_ACRONYMS.items():
            pattern = r"\b" + re.escape(acronym) + r"\b"
            text = re.sub(pattern, spoken, text)
        return text

    @classmethod
    def normalize(cls, raw_text: str) -> Dict[str, Any]:
        """
        Full text normalization pipeline with granular tracking.
        Returns normalized text along with NSW transformations log.
        """
        original = raw_text
        step1 = cls.normalize_abbreviations(original)
        step2 = cls.normalize_currencies(step1)
        step3 = cls.normalize_percentages(step2)
        step4 = cls.normalize_units(step3)
        step5 = cls.normalize_times(step4)
        step6 = cls.normalize_ordinals(step5)
        step7 = cls.normalize_numbers(step6)
        normalized = cls.normalize_acronyms(step7)

        normalized = re.sub(r"\s+", " ", normalized).strip()

        return {
            "original_text": original,
            "normalized_text": normalized,
            "changes_detected": original != normalized
        }


class G2PPhonemizer:
    """
    Grapheme-to-Phoneme (G2P) phonetic transcription engine.
    Translates orthographic words into IPA / ARPAbet phonetic representations
    with context-aware homograph disambiguation and syllable stress markers.
    """

    LEXICON = {
        "hello": {"ipa": "həˈloʊ", "arpabet": "HH AH0 L OW1", "syllables": ["hel", "lo"], "stress": 2},
        "mira": {"ipa": "ˈmɪərə", "arpabet": "M IH1 R AH0", "syllables": ["mi", "ra"], "stress": 1},
        "voice": {"ipa": "vɔɪs", "arpabet": "V OY1 S", "syllables": ["voice"], "stress": 1},
        "cloning": {"ipa": "ˈkloʊnɪŋ", "arpabet": "K L OW1 N IH0 NG", "syllables": ["clon", "ing"], "stress": 1},
        "neural": {"ipa": "ˈnjʊərəl", "arpabet": "N Y UH1 R AH0 L", "syllables": ["neu", "ral"], "stress": 1},
        "speech": {"ipa": "spiːtʃ", "arpabet": "S P IY1 CH", "syllables": ["speech"], "stress": 1},
        "synthesis": {"ipa": "ˈsɪnθəsɪs", "arpabet": "S IH1 N TH AH0 S IH0 S", "syllables": ["syn", "the", "sis"], "stress": 1},
        "acoustic": {"ipa": "əˈkuːstɪk", "arpabet": "AH0 K UW1 S T IH0 K", "syllables": ["a", "cous", "tic"], "stress": 2},
        "model": {"ipa": "ˈmɒdəl", "arpabet": "M AA1 D AH0 L", "syllables": ["mod", "el"], "stress": 1},
        "vocoder": {"ipa": "voʊˈkoʊdər", "arpabet": "V OW0 K OW1 D ER0", "syllables": ["vo", "co", "der"], "stress": 2},
        "spectrogram": {"ipa": "ˈspɛktrəɡræm", "arpabet": "S P EH1 K T R AH0 G R AE2 M", "syllables": ["spec", "tro", "gram"], "stress": 1},
        "frequency": {"ipa": "ˈfriːkwənsi", "arpabet": "F R IY1 K W AH0 N S IY0", "syllables": ["fre", "quen", "cy"], "stress": 1},
        "harmonic": {"ipa": "hɑːrˈmɒnɪk", "arpabet": "HH AA0 R M AA1 N IH0 K", "syllables": ["har", "mon", "ic"], "stress": 2},
        "formant": {"ipa": "ˈfɔːrmənt", "arpabet": "F AO1 R M AH0 N T", "syllables": ["for", "mant"], "stress": 1},
        "timbre": {"ipa": "ˈtæmbər", "arpabet": "T AE1 M B ER0", "syllables": ["tim", "bre"], "stress": 1},
        "cadence": {"ipa": "ˈkeɪdəns", "arpabet": "K EY1 D AH0 N S", "syllables": ["ca", "dence"], "stress": 1},
        "prosody": {"ipa": "ˈprɒsədi", "arpabet": "P R AA1 S AH0 D IY0", "syllables": ["pros", "o", "dy"], "stress": 1},
        "waveform": {"ipa": "ˈweɪvfɔːrm", "arpabet": "W EY1 V F AO2 R M", "syllables": ["wave", "form"], "stress": 1},
        "interviewer": {"ipa": "ˈɪntərvjuːər", "arpabet": "IH1 N T ER0 V Y UW2 ER0", "syllables": ["in", "ter", "view", "er"], "stress": 1},
        "candidate": {"ipa": "ˈkændɪdeɪt", "arpabet": "K AE1 N D AH0 D EY2 T", "syllables": ["can", "di", "date"], "stress": 1},
        "system": {"ipa": "ˈsɪstəm", "arpabet": "S IH1 S T AH0 M", "syllables": ["sys", "tem"], "stress": 1},
        "pipeline": {"ipa": "ˈpaɪplaɪn", "arpabet": "P AY1 P L AY2 N", "syllables": ["pipe", "line"], "stress": 1},
        "architecture": {"ipa": "ˈɑːrkɪtɛktʃər", "arpabet": "AA1 R K AH0 T EH2 K CH ER0", "syllables": ["ar", "chi", "tec", "ture"], "stress": 1},
        "hundred": {"ipa": "ˈhʌndrəd", "arpabet": "HH AH1 N D R AH0 D", "syllables": ["hun", "dred"], "stress": 1},
        "twenty": {"ipa": "ˈtwɛnti", "arpabet": "T W EH1 N T IY0", "syllables": ["twen", "ty"], "stress": 1},
        "six": {"ipa": "sɪks", "arpabet": "S IH1 K S", "syllables": ["six"], "stress": 1},
        "dollars": {"ipa": "ˈdɒlərz", "arpabet": "D AA1 L ER0 Z", "syllables": ["dol", "lars"], "stress": 1},
        "percent": {"ipa": "pərˈsɛnt", "arpabet": "P ER0 S EH1 N T", "syllables": ["per", "cent"], "stress": 2},
        "doctor": {"ipa": "ˈdɒktər", "arpabet": "D AA1 K T ER0", "syllables": ["doc", "tor"], "stress": 1},
    }

    VOWEL_MAP = {
        "a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɒ", "u": "ʌ",
        "ai": "eɪ", "ay": "eɪ", "ea": "iː", "ee": "iː",
        "oo": "uː", "ou": "aʊ", "ow": "aʊ", "oi": "ɔɪ", "oy": "ɔɪ"
    }

    CONSONANT_MAP = {
        "sh": "ʃ", "ch": "tʃ", "th": "θ", "ph": "f", "ck": "k",
        "ng": "ŋ", "wh": "w", "qu": "kw", "c": "k", "j": "dʒ",
        "x": "ks", "y": "j"
    }

    @classmethod
    def rule_based_phonemize(cls, word: str) -> Dict[str, Any]:
        """Approximates IPA phonemes for out-of-vocabulary words using linguistic heuristics."""
        w = word.lower()
        ipa_parts = []
        i = 0
        while i < len(w):
            if i + 1 < len(w) and w[i:i+2] in cls.CONSONANT_MAP:
                ipa_parts.append(cls.CONSONANT_MAP[w[i:i+2]])
                i += 2
            elif i + 1 < len(w) and w[i:i+2] in cls.VOWEL_MAP:
                ipa_parts.append(cls.VOWEL_MAP[w[i:i+2]])
                i += 2
            elif w[i] in cls.VOWEL_MAP:
                ipa_parts.append(cls.VOWEL_MAP[w[i]])
                i += 1
            elif w[i] in cls.CONSONANT_MAP:
                ipa_parts.append(cls.CONSONANT_MAP[w[i]])
                i += 1
            else:
                ipa_parts.append(w[i])
                i += 1

        ipa_str = "".join(ipa_parts)
        ipa_with_stress = "ˈ" + ipa_str if ipa_str else ""
        return {
            "ipa": ipa_with_stress,
            "arpabet": " ".join([c.upper() for c in ipa_parts]),
            "syllables": [w],
            "stress": 1,
            "heuristic": True
        }

    @classmethod
    def phonemize_word(cls, word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> Dict[str, Any]:
        """Translates a single word into phonetic notation with homograph disambiguation."""
        clean = re.sub(r"[^\w]", "", word.lower())
        if not clean:
            return {"word": word, "ipa": "", "arpabet": "", "syllables": [], "stress": 0}

        # Check for context-sensitive homographs first
        homo = HomographDisambiguator.disambiguate(clean, prev_word=prev_word, next_word=next_word)
        if homo:
            return {
                "word": word,
                "ipa": homo["ipa"],
                "arpabet": homo["arpabet"],
                "syllables": [clean],
                "stress": 1,
                "heuristic": False,
                "homograph": True,
                "homograph_rule": homo["rule"],
                "sense": homo.get("sense")
            }

        if clean in cls.LEXICON:
            entry = dict(cls.LEXICON[clean])
            entry["word"] = word
            entry["heuristic"] = False
            return entry

        entry = cls.rule_based_phonemize(clean)
        entry["word"] = word
        return entry

    @classmethod
    def phonemize_sentence(cls, sentence: str) -> Dict[str, Any]:
        """
        Translates a full sentence into phonetic tokens with contextual homograph resolution.
        Preserves punctuation for prosodic pause boundaries.
        """
        tokens = re.findall(r"\w+|[^\w\s]", sentence)
        results = []
        full_ipa = []
        full_arpabet = []
        homographs_found = []

        words_only = [t for t in tokens if not re.match(r"[^\w\s]", t)]

        for i, token in enumerate(tokens):
            if re.match(r"[^\w\s]", token):
                results.append({"word": token, "type": "punctuation", "ipa": token, "arpabet": token})
                full_ipa.append(token)
                full_arpabet.append(token)
            else:
                # Find previous and next word for context
                w_idx = words_only.index(token) if token in words_only else -1
                prev_w = words_only[w_idx - 1] if w_idx > 0 else None
                next_w = words_only[w_idx + 1] if 0 <= w_idx < len(words_only) - 1 else None

                phon = cls.phonemize_word(token, prev_word=prev_w, next_word=next_w)
                results.append(phon)
                if phon.get("ipa"):
                    full_ipa.append(phon["ipa"])
                if phon.get("arpabet"):
                    full_arpabet.append(phon["arpabet"])
                if phon.get("homograph"):
                    homographs_found.append({
                        "word": token,
                        "ipa": phon["ipa"],
                        "rule": phon["homograph_rule"],
                        "sense": phon.get("sense")
                    })

        return {
            "input_sentence": sentence,
            "tokens": results,
            "ipa_string": " ".join(full_ipa),
            "arpabet_string": " ".join(full_arpabet),
            "token_count": len(results),
            "homographs_disambiguated": homographs_found
        }
