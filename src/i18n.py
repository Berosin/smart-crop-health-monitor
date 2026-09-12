"""Regional language support — Tamil.

Real farmers in this app's target use case (Tamil Nadu) often don't read
English fluently. A diagnosis and recommendation only helps someone if
they can actually read it — so this module lets the app's core
farmer-facing output (disease name, severity, recommendation) render in
Tamil, toggled from the sidebar, alongside the existing English.

Scope, deliberately
---------------------
This translates the FIXED, canned strings that matter most for a farmer
making a decision: disease names, severity labels/actions, and the
disease recommendation texts (config-style lookup tables, not live text
generation) — plus the same set carried through into Field Scan and PDF
reports, since a shared PDF is exactly the kind of artifact a farmer
without reliable English needs in their own language.

It deliberately does NOT attempt to translate the whole UI (charts, help
text, dashboard analytics, technical explanations) — those are either
dynamically generated (interpolated with live numbers, which a static
dictionary can't safely translate without a live translation API) or
lower-priority for the "can a farmer act on this" narrative. See the
project README/conversation history for the reasoning: a static
dictionary for a bounded set of real strings, no live API, translated
once and reviewed — not "translate everything."

Terminology sourcing
-----------------------
Disease terminology was cross-checked against Tamil Nadu Agricultural
University's own Tamil-language extension pages (agritech.tnau.ac.in/ta)
rather than invented — e.g. "கருகல் நோய்" (blight), "குலை நோய்" (blast,
with the confirmed compound "கழுத்து குலை நோய்" for neck blast), and
"துரு நோய்" (rust, with "பழுப்பு துருநோய்" confirmed for brown/leaf rust
on wheat) all come directly from TNAU's own published material. Compound
names not directly published by TNAU (e.g. this app's specific corn
disease classes) are constructed from those same verified roots. As with
any bounded translation effort, a native Tamil agricultural extension
reviewer should sign off before this is relied on for real farmer
decisions at scale — this gets the terminology right by sourcing, not by
claiming authority the codebase doesn't have.
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {"en": "English", "ta": "தமிழ்"}
DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Crop names
# ---------------------------------------------------------------------------
CROP_NAMES_TA: dict[str, str] = {
    "Tomato": "தக்காளி",
    "Potato": "உருளைக்கிழங்கு",
    "Rice": "நெல்",
    "Wheat": "கோதுமை",
    "Corn": "மக்காச்சோளம்",
}

# ---------------------------------------------------------------------------
# Disease display names (source-verified against TNAU where possible —
# see module docstring)
# ---------------------------------------------------------------------------
DISEASE_NAMES_TA: dict[str, str] = {
    "Healthy": "ஆரோக்கியமானது",
    "Early_Blight": "ஆரம்பநிலை கருகல் நோய்",
    "Late_Blight": "தாமதமான கருகல் நோய்",
    "Brown_Spot": "பழுப்பு புள்ளி நோய்",
    "Leaf_Blast": "இலை குலை நோய்",
    "Neck_Blast": "கழுத்து குலை நோய்",
    "Brown_Rust": "பழுப்பு துரு நோய்",
    "Yellow_Rust": "மஞ்சள் துரு நோய்",
    "Common_Rust": "பொதுவான துரு நோய்",
    "Gray_Leaf_Spot": "சாம்பல் நிற இலைப் புள்ளி நோய்",
    "Northern_Leaf_Blight": "வடக்கு மக்காச்சோள இலைக் கருகல் நோய்",
}

# ---------------------------------------------------------------------------
# Severity levels + the action-phrases pages/disease.py's SEVERITY_META uses
# ---------------------------------------------------------------------------
SEVERITY_LABELS_TA: dict[str, str] = {
    "None": "இல்லை",
    "Mild": "லேசானது",
    "Moderate": "மிதமானது",
    "High": "அதிகம்",
    "Unknown": "தெரியவில்லை",
}

SEVERITY_ACTION_TA: dict[str, str] = {
    "No action needed": "நடவடிக்கை தேவையில்லை",
    "Monitor closely": "உன்னிப்பாகக் கண்காணிக்கவும்",
    "Treat promptly": "உடனடியாக சிகிச்சை அளிக்கவும்",
    "Intervene urgently": "அவசரமாகத் தலையிடவும்",
    "Unknown": "தெரியவில்லை",
}

# ---------------------------------------------------------------------------
# Disease recommendation texts — full translations of
# pages/disease.py's RECOMMENDATION_MAP. Chemical/treatment names are
# transliterated into Tamil script (e.g. மேன்கோசெப் for mancozeb,
# டிரைசைக்ளோசோல் for tricyclazole), matching how TNAU's own Tamil
# extension material writes them, rather than left in Latin script.
# ---------------------------------------------------------------------------
RECOMMENDATION_TA: dict[str, str] = {
    "Healthy":
        "பயிர் ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பையும் சீரான "
        "நீர்ப்பாசனத்தையும் தொடரவும்.",
    "Early_Blight":
        "ஆரம்பநிலை கருகல் நோய் கண்டறியப்பட்டது. பாதிக்கப்பட்ட இலைகளை "
        "அகற்றவும், காப்பர் அடிப்படையிலான பூசணக்கொல்லி தெளிக்கவும், "
        "செடிகளுக்கு இடையே காற்றோட்டத்தை மேம்படுத்தவும்.",
    "Late_Blight":
        "தாமதமான கருகல் நோய் கண்டறியப்பட்டது. அவசரம்: பாதிக்கப்பட்ட "
        "செடிகளை அழிக்கவும், மேன்கோசெப் தெளிக்கவும், ஈரமான நேரத்தில் "
        "வயலில் வேலை செய்வதைத் தவிர்க்கவும்.",
    "Brown_Spot":
        "பழுப்பு புள்ளி நோய் கண்டறியப்பட்டது. வயல் ஊட்டச்சத்தை "
        "(குறிப்பாக மணிச்சத்தை) மேம்படுத்தவும், பரிந்துரைக்கப்பட்ட "
        "பூசணக்கொல்லி (எ.கா. புரோபிகோனசோல்) தெளிக்கவும், நீர் "
        "பற்றாக்குறையைத் தவிர்க்கவும்.",
    "Leaf_Blast":
        "இலை குலை நோய் கண்டறியப்பட்டது. டிரைசைக்ளோசோல் அடிப்படையிலான "
        "பூசணக்கொல்லியைத் தெளிக்கவும், அதிக தழைச்சத்தைத் தவிர்க்கவும், "
        "வயலில் சரியான நீர் மேலாண்மையை பராமரிக்கவும்.",
    "Neck_Blast":
        "கழுத்து குலை நோய் கண்டறியப்பட்டது. அவசரம்: கதிர் வெளிவரும் "
        "ஆரம்ப நிலையில் பூசணக்கொல்லி தெளிக்கவும், அடர்த்தியான நடவை "
        "தவிர்க்கவும், இது மகசூலை கணிசமாகக் குறைக்கக்கூடும் என்பதால் "
        "உன்னிப்பாகக் கண்காணிக்கவும்.",
    "Brown_Rust":
        "பழுப்பு துரு நோய் கண்டறியப்பட்டது. ட்ரையசோல் அடிப்படையிலான "
        "பூசணக்கொல்லியைத் தெளிக்கவும், பரவலுக்கு உகந்த சூடான ஈரப்பதமான "
        "சூழலைக் கண்காணிக்கவும், அடுத்த சாகுபடிக்கு எதிர்ப்புத்திறன் "
        "கொண்ட ரகத்தைக் கருதவும்.",
    "Yellow_Rust":
        "மஞ்சள் துரு நோய் கண்டறியப்பட்டது. உடனடியாக ட்ரையசோல் "
        "அடிப்படையிலான பூசணக்கொல்லியைத் தெளிக்கவும், குளிர்ந்த "
        "ஈரப்பதமான சூழலை உன்னிப்பாகக் கண்காணிக்கவும் (இந்த துரு நோய் "
        "குளிர் காலநிலையில் வேகமாகப் பரவும்), அருகிலுள்ள வயல்களையும் "
        "பரிசோதிக்கவும்.",
    "Common_Rust":
        "பொதுவான துரு நோய் கண்டறியப்பட்டது. தீவிரமாக இருந்தால் "
        "ஸ்ட்ரோபிலூரின் அல்லது ட்ரையசோல் பூசணக்கொல்லியைத் தெளிக்கவும், "
        "எதிர்கால சாகுபடிக்கு துருவை எதிர்க்கும் கலப்பின ரகங்களைத் "
        "தேர்வு செய்யவும்.",
    "Gray_Leaf_Spot":
        "சாம்பல் நிற இலைப் புள்ளி நோய் கண்டறியப்பட்டது. ஒரு பருவத்திற்கு "
        "மக்காச்சோளம்/பயிர் எச்சத்திலிருந்து பயிர் சுழற்சி செய்யவும், "
        "நோய் அழுத்தம் அதிகமாக இருந்தால் இலை வழி பூசணக்கொல்லி "
        "தெளிக்கவும், வயல் காற்றோட்டத்தை மேம்படுத்தவும்.",
    "Northern_Leaf_Blight":
        "வடக்கு மக்காச்சோள இலைக் கருகல் நோய் கண்டறியப்பட்டது. குறிப்பாக "
        "பூக்கும் நிலைக்கு முன், உடனடியாக இலை வழி பூசணக்கொல்லி "
        "தெளிக்கவும், மகசூல் இழப்பு கடுமையாக இருக்கக்கூடும் என்பதால் "
        "அடுத்த பருவத்திற்கு எதிர்ப்புத்திறன் கொண்ட கலப்பின ரகங்களைக் "
        "கருதவும்.",
}

# ---------------------------------------------------------------------------
# A small set of key farmer-facing UI labels (section headers, status
# words) — NOT a full UI translation, just the labels that surround the
# translated content above so it doesn't look out of place.
# ---------------------------------------------------------------------------
UI_LABELS_TA: dict[str, str] = {
    "Detected condition": "கண்டறியப்பட்ட நிலை",
    "Confidence": "நம்பகத்தன்மை",
    "model output": "மாதிரி வெளியீடு",
    "Severity": "தீவிரம்",
    "Threshold": "வரம்பு",
    "cutoff for reliable result": "நம்பகமான முடிவுக்கான வரம்பு",
    "Recommendation": "பரிந்துரை",
    "Confidence breakdown by class": "வகைவாரியான நம்பகத்தன்மை பிரிவு",
    "Estimated yield loss if untreated": "சிகிச்சை அளிக்கவில்லை எனில் எதிர்பார்க்கப்படும் மகசூல் இழப்பு",
    "Estimated Yield Loss If Untreated": "சிகிச்சை அளிக்கவில்லை எனில் எதிர்பார்க்கப்படும் மகசூல் இழப்பு",
    "Healthy": "ஆரோக்கியமானது",
    "Disease detected": "நோய் கண்டறியப்பட்டது",
    "Crop": "பயிர்",
    "Prediction": "முன்னறிவிப்பு",
    "Status": "நிலை",
    # PDF report labels (src/report_generator.py)
    "Crop Disease Diagnostic Report": "பயிர் நோய் கண்டறிதல் அறிக்கை",
    "Field Scan Report": "வயல் ஆய்வு அறிக்கை",
    "Image": "படம்",
    "Analyzed leaf image": "பகுப்பாய்வு செய்யப்பட்ட இலைப் படம்",
    "Grad-CAM: what drove this prediction": "Grad-CAM: இந்த முடிவுக்கான காரணம்",
    "leaves scanned": "இலைகள் ஆய்வு செய்யப்பட்டன",
    "Photos scanned": "ஆய்வு செய்யப்பட்ட புகைப்படங்கள்",
    "Dominant disease": "முதன்மை நோய்",
    "None detected": "எதுவும் கண்டறியப்படவில்லை",
    "Field health score": "வயல் ஆரோக்கிய மதிப்பெண்",
    "Disease Breakdown": "நோய் பிரிவு",
    "Disease": "நோய்",
    "Severity Breakdown": "தீவிர பிரிவு",
    "Individual Leaves": "தனிப்பட்ட இலைகள்",
    "Leaves": "இலைகள்",
    "Photo": "புகைப்படம்",
    "Confidence": "நம்பகத்தன்மை",
    "Note": "குறிப்பு",
    "Generated": "உருவாக்கப்பட்டது",
}

# ---------------------------------------------------------------------------
# Sidebar navigation labels (utils/ui.py's PAGES)
# ---------------------------------------------------------------------------
NAV_LABELS_TA: dict[str, str] = {
    "Home": "முகப்பு",
    "Dashboard": "கட்டுப்பாட்டு பலகை",
    "Disease Detection": "நோய் கண்டறிதல்",
    "Field Scan": "வயல் ஆய்வு",
    "Outbreak Alerts": "பரவல் எச்சரிக்கைகள்",
    "Environmental Analysis": "சுற்றுச்சூழல் பகுப்பாய்வு",
    "Crop Health Analysis": "பயிர் ஆரோக்கிய பகுப்பாய்வு",
    "Analysis History": "பகுப்பாய்வு வரலாறு",
    "About Project": "திட்டம் பற்றி",
}

# ---------------------------------------------------------------------------
# Health-status words (utils/ui.py's health_score_card) and risk levels
# (utils/ui.py's RISK_LEVELS, used app-wide: environment risk, outbreak
# risk, field scan status, etc.)
# ---------------------------------------------------------------------------
HEALTH_STATUS_TA: dict[str, str] = {
    "Excellent": "சிறந்தது",
    "Good": "நல்லது",
    "Poor": "மோசமானது",
    "Critical": "ஆபத்தானது",
    "Overall health score": "மொத்த ஆரோக்கிய மதிப்பெண்",
    "Grade": "தரம்",
}

RISK_LEVEL_TA: dict[str, str] = {
    "Optimal": "உகந்தது",
    "Low": "குறைவு",
    "Moderate": "மிதமானது",
    "High": "அதிகம்",
    "Critical": "ஆபத்தானது",
    "Watch": "கவனி",
    "Elevated": "உயர்ந்தது",
    "Insufficient data": "போதிய தரவு இல்லை",
    "Unknown": "தெரியவில்லை",
    # Environmental/crop-health-analysis status bands (src/health_engine.py)
    "Healthy": "ஆரோக்கியமானது",
    "At Risk": "ஆபத்தில் உள்ளது",
}

MISC_TA: dict[str, str] = {
    "Software-only demo": "மென்பொருள் மட்டும் கொண்ட மாதிரி",
    "No hardware required": "வன்பொருள் தேவையில்லை",
    "Agriculture-themed demo build": "விவசாய கருப்பொருள் மாதிரி பதிப்பு",
    "Navigate": "வழிசெலுத்து",
    "AI-powered crop disease detection and health scoring": "AI-இயங்கும் பயிர் நோய் கண்டறிதல் மற்றும் ஆரோக்கிய மதிப்பீடு",
}

# ---------------------------------------------------------------------------
# Environmental Analysis (pages/environment.py + src/environment_model.py)
# ---------------------------------------------------------------------------
ENV_FACTOR_LABELS_TA: dict[str, str] = {
    "Temperature": "வெப்பநிலை",
    "Humidity": "ஈரப்பதம்",
    "Soil moisture": "மண் ஈரப்பதம்",
    "Rainfall": "மழைப்பொழிவு",
}

ENV_STATUS_TA: dict[str, str] = {
    "Optimal": "உகந்தது",
    "Low": "குறைவு",
    "High": "அதிகம்",
    "Extreme": "தீவிரமானது",
}

ENV_NOTE_TA: dict[str, str] = {
    "Within ideal range": "சிறந்த வரம்பிற்குள்",
    "Outside safe range": "பாதுகாப்பான வரம்பிற்கு வெளியே",
    "Below ideal": "சிறந்ததை விட குறைவு",
    "Above ideal": "சிறந்ததை விட அதிகம்",
}

# Per-factor advice templates — pages/environment.py's _advice() and
# src/environment_model.py's TIPS share the same 12 templates (temperature/
# humidity/soil_moisture/rainfall × Low/High/Extreme). One Tamil table,
# reused by both. {crop} stays as a format placeholder — filled in with
# tr_crop() at call time so the crop name is translated too.
ENV_TIPS_TA: dict[str, dict[str, str]] = {
    "temperature": {
        "Low": "வரிசை மூடிகளைப் பயன்படுத்தவும் அல்லது மண்ணை வெப்பப்படுத்தவும் — குளிர்ச்சி {crop} வளர்ச்சியை மெதுவாக்கும்.",
        "High": "நிழல் துணி வழங்கவும் அல்லது குளிர்ந்த மாலை நேரத்தில் நீர்ப்பாசனம் செய்யவும் — இது {crop} மீதான வெப்ப அழுத்தத்தைக் குறைக்கும்.",
        "Extreme": "வெப்பநிலை பாதுகாப்பான வரம்பிற்கு வெளியே உள்ளது — {crop}-ஐ பாதுகாக்கவும் அல்லது முக்கியமான வயல் பணிகளை தாமதப்படுத்தவும்.",
    },
    "humidity": {
        "Low": "ஈரப்பதம் குறைவாக உள்ளது; ஈரப்பத இழப்பையும் இலை சுருளையும் குறைக்க {crop} சுற்றிலும் மல்ச் செய்யவும்.",
        "High": "அதிக ஈரப்பதம் {crop}-க்கு பூஞ்சை நோய் அபாயத்தை அதிகரிக்கிறது; காற்றோட்டத்தை மேம்படுத்தி மேலிருந்து நீர் பாய்ச்சுவதைத் தவிர்க்கவும்.",
        "Extreme": "ஈரப்பத அளவு தீவிரமாக உள்ளது — நோய் அல்லது வாடலுக்காக {crop}-ஐ உன்னிப்பாகக் கண்காணிக்கவும்.",
    },
    "soil_moisture": {
        "Low": "{crop}-க்கு மண் வறண்டுள்ளது; நீர்ப்பாசனத்தை திட்டமிட்டு வேர் பகுதிகளை சரிபார்க்கவும்.",
        "High": "{crop}-க்கு மண் நீரில் மூழ்கியுள்ளது; வடிகால் மேம்படுத்தி நீர்ப்பாசனத்தை தற்காலிகமாக நிறுத்தவும்.",
        "Extreme": "{crop}-க்கு மண் ஈரப்பதம் அளவிற்கு அப்பாற்பட்டது; வடிகால் அல்லது நீர்ப்பாசனத்தை உடனடியாக சரிசெய்யவும்.",
    },
    "rainfall": {
        "Low": "மழைப்பொழிவு குறைவாக உள்ளது; {crop}-க்கு நீர்ப்பாசனம் மூலம் துணைபுரியவும்.",
        "High": "அதிக மழை எதிர்பார்க்கப்படுகிறது; {crop} சுற்றிலும் வடிகால் மற்றும் ஊட்டச்சத்து வடிதல் குறித்து கவனமாக இருங்கள்; வயல் வடிகால் உறுதி செய்யவும்.",
        "Extreme": "{crop}-க்கு அதிக மழைப்பொழிவு; தாழ்வான பகுதிகளைப் பாதுகாத்து நீர் தேங்கியிருப்பதை சரிபார்க்கவும்.",
    },
}

ENV_ADVICE_ALL_OPTIMAL_TA = "{crop}-க்கு அனைத்து சுற்றுச்சூழல் காரணிகளும் சிறந்த வரம்பிற்குள் உள்ளன. தற்போதைய நடைமுறைகளைத் தொடர்ந்து கண்காணிக்கவும்."

ENV_UI_TA: dict[str, str] = {
    "Environmental Analysis": "சுற்றுச்சூழல் பகுப்பாய்வு",
    "Log environmental conditions and assess crop risk with the trained model.":
        "சுற்றுச்சூழல் நிலைமைகளை பதிவு செய்து பயிற்சி பெற்ற மாதிரி மூலம் பயிர் ஆபத்தை மதிப்பிடவும்.",
    "Enter environmental readings": "சுற்றுச்சூழல் அளவீடுகளை உள்ளிடவும்",
    "Crop type": "பயிர் வகை",
    "Data source": "தரவு மூலம்",
    "Manual Entry": "கைமுறை உள்ளீடு",
    "Live Weather": "நேரடி வானிலை",
    "Live Weather auto-fills temperature, humidity, and rainfall "
    "from OpenWeatherMap for a location you choose. Soil moisture "
    "always needs a manual reading — no weather API measures it. "
    "Manual Entry (the default) needs nothing beyond this page.":
        "நேரடி வானிலை நீங்கள் தேர்ந்தெடுக்கும் இடத்திற்கான OpenWeatherMap தரவிலிருந்து "
        "வெப்பநிலை, ஈரப்பதம், மழைப்பொழிவை தானாக நிரப்பும். மண் ஈரப்பதத்திற்கு எப்போதும் "
        "கைமுறை அளவீடு தேவை — எந்த வானிலை API-யும் அதை அளவிடாது. கைமுறை உள்ளீடு "
        "(இயல்புநிலை) இந்தப் பக்கத்தைத் தாண்டி வேறு எதுவும் தேவையில்லை.",
    "Not available from weather data — enter your own reading.":
        "வானிலை தரவிலிருந்து கிடைக்கவில்லை — உங்கள் சொந்த அளவீட்டை உள்ளிடவும்.",
    "Assess environment": "சுற்றுச்சூழலை மதிப்பிடவும்",
    "Input summary": "உள்ளீட்டு சுருக்கம்",
    "What will be sent to the risk model.": "ஆபத்து மாதிரிக்கு அனுப்பப்படும் தரவு.",
    "Live weather for": "நேரடி வானிலை",
    "fetched": "பெறப்பட்டது",
    "Click **Assess environment** to run the trained risk model "
    "and see per-factor statuses, an overall risk level, and a visualization.":
        "பயிற்சி பெற்ற ஆபத்து மாதிரியை இயக்கவும் ஒவ்வொரு காரணியின் நிலை, ஒட்டுமொத்த "
        "ஆபத்து அளவு மற்றும் ஒரு காட்சிப்படுத்தலைப் பார்க்கவும் **சுற்றுச்சூழலை மதிப்பிடவும்** "
        "என்பதைக் கிளிக் செய்யவும்.",
    "Environmental risk model not found.": "சுற்றுச்சூழல் ஆபத்து மாதிரி கிடைக்கவில்லை.",
    "Train it first with": "முதலில் இதன் மூலம் பயிற்சி செய்யவும்",
    "Assessing environmental risk failed unexpectedly. Please try "
    "again. If the problem continues, contact the app maintainer.":
        "சுற்றுச்சூழல் ஆபத்தை மதிப்பிடுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Factor visualization": "காரணி காட்சிப்படுத்தல்",
    "Overall risk level": "ஒட்டுமொத்த ஆபத்து அளவு",
    "Env. health score": "சுற்றுச்சூழல் ஆரோக்கிய மதிப்பெண்",
    "model confidence": "மாதிரி நம்பகத்தன்மை",
    "Factor": "காரணி",
    "Value": "மதிப்பு",
    "Ideal range": "சிறந்த வரம்பு",
    "Status": "நிலை",
    "Note": "குறிப்பு",
    "Recommendations": "பரிந்துரைகள்",
    "Location": "இடம்",
    "OpenWeatherMap API key": "OpenWeatherMap API விசை",
    "City name, optionally with a country code, e.g. 'Chennai, IN'.":
        "நகர பெயர், விருப்பமாக நாட்டுக் குறியீடுடன், எ.கா. 'Chennai, IN'.",
    "No OpenWeatherMap API key configured yet. "
    "Get a free one at":
        "இதுவரை OpenWeatherMap API விசை அமைக்கப்படவில்லை. இங்கே இலவசமாகப் பெறவும்",
    "and paste it above, or set":
        "மேலே ஒட்டவும், அல்லது அமைக்கவும்",
    "as an environment variable / in":
        "சூழல் மாறியாக / இதில்",
    "so you don't have to re-enter it.":
        "இதனால் மீண்டும் உள்ளிட வேண்டியதில்லை.",
    "Fetch current weather": "நேரடி வானிலையைப் பெறவும்",
    "Fetching weather failed unexpectedly. Please try again, or "
    "switch to Manual Entry above.":
        "வானிலையைப் பெறுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும் "
        "அல்லது மேலே உள்ள கைமுறை உள்ளீட்டிற்கு மாறவும்.",
    "humidity": "ஈரப்பதம்",
    "Short-term disease risk forecast": "குறுகிய கால நோய் ஆபத்து முன்னறிவிப்பு",
    "Runs the same trained risk model against the next few days' forecast "
    "instead of a single reading. Soil moisture is held at your current "
    "value above": "ஒற்றை அளவீட்டிற்குப் பதிலாக அடுத்த சில நாட்களின் முன்னறிவிப்பிற்கு எதிராக "
    "அதே பயிற்சி பெற்ற ஆபத்து மாதிரியை இயக்குகிறது. மண் ஈரப்பதம் மேலே உள்ள உங்கள் தற்போதைய "
    "மதிப்பில் வைக்கப்பட்டுள்ளது",
    "for every day — no weather API forecasts soil moisture.":
        "ஒவ்வொரு நாளும் — எந்த வானிலை API-யும் மண் ஈரப்பதத்தை முன்னறிவிக்காது.",
    "Fetch current weather above first (needs a location and API key).":
        "முதலில் மேலே நேரடி வானிலையைப் பெறவும் (ஒரு இடமும் API விசையும் தேவை).",
    "Load 5-day risk forecast": "5-நாள் ஆபத்து முன்னறிவிப்பை ஏற்றவும்",
    "Building the forecast failed unexpectedly. Please try again.":
        "முன்னறிவிப்பை உருவாக்குவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.",
    "looks highest-risk:": "அதிக ஆபத்துடையதாகத் தெரிகிறது:",
    "rain": "மழை",
    "Ideal band": "சிறந்த வரம்பு",
    "Current": "தற்போதைய",
    "Readings are normalized 0–1 against each factor's safe band "
    "for **{crop}**. The shaded green zone is the ideal range.":
        "ஒவ்வொரு காரணியின் பாதுகாப்பான வரம்பிற்கு எதிராக அளவீடுகள் 0–1 ஆக "
        "இயல்பாக்கப்பட்டுள்ளன **{crop}**-க்கு. நிழலிடப்பட்ட பச்சை பகுதி சிறந்த வரம்பு.",
    "Analysis saved to database (ID:": "பகுப்பாய்வு தரவுத்தளத்தில் சேமிக்கப்பட்டது (ID:",
    "Saved": "சேமிக்கப்பட்டது",
    "Save Analysis": "பகுப்பாய்வை சேமிக்கவும்",
    "Env. health score": "சுற்றுச்சூழல் ஆரோக்கிய மதிப்பெண்",
    "model confidence": "மாதிரி நம்பகத்தன்மை",
}

# ---------------------------------------------------------------------------
# src/health_engine.py — dynamic explanation/recommendation sentence
# templates. {placeholders} are preserved and filled in by health_engine.py
# itself (values, not English words, so no further translation needed for
# those parts).
# ---------------------------------------------------------------------------
HEALTH_ENGINE_TA: dict[str, str] = {
    "Model predicts '{disease}' with {confidence}% confidence — no disease penalty applied.":
        "மாதிரி '{disease}' என்று {confidence}% நம்பகத்தன்மையுடன் கணிக்கிறது — நோய் அபராதம் "
        "எதுவும் பொருந்தவில்லை.",
    "'{disease}' detected at {severity} severity with {confidence}% confidence — "
    "both severity and confidence increase risk.":
        "'{disease}' {severity} தீவிரத்தில் {confidence}% நம்பகத்தன்மையுடன் கண்டறியப்பட்டது — "
        "தீவிரமும் நம்பகத்தன்மையும் ஆபத்தை அதிகரிக்கின்றன.",
    "Environmental risk classified as {risk}{conf_text}, scored using {basis}.":
        "சுற்றுச்சூழல் ஆபத்து {risk} என வகைப்படுத்தப்பட்டது{conf_text}, {basis} பயன்படுத்தி மதிப்பிடப்பட்டது.",
    "a probability-weighted average across all predicted risk classes":
        "கணிக்கப்பட்ட அனைத்து ஆபத்து வகைகளிலும் நிகழ்தகவு-எடையிடப்பட்ட சராசரி",
    "the single predicted risk class": "ஒற்றை கணிக்கப்பட்ட ஆபத்து வகை",
    "{disease_explanation} This contributes a disease score of {disease_score}/100. "
    "{environmental_explanation} This contributes an environmental score of "
    "{environmental_score}/100. Weighted {disease_weight}% disease / "
    "{environmental_weight}% environment, the overall health score is "
    "{health_score}/100, classified as '{status}'.":
        "{disease_explanation} இது {disease_score}/100 நோய் மதிப்பெண்ணை அளிக்கிறது. "
        "{environmental_explanation} இது {environmental_score}/100 சுற்றுச்சூழல் "
        "மதிப்பெண்ணை அளிக்கிறது. {disease_weight}% நோய் / {environmental_weight}% "
        "சுற்றுச்சூழலுக்கு எடையிடப்பட்டு, ஒட்டுமொத்த ஆரோக்கிய மதிப்பெண் "
        "{health_score}/100, '{status}' என வகைப்படுத்தப்பட்டுள்ளது.",
    # FALLBACK_DISEASE_ADVICE
    "Crop appears healthy. Maintain regular monitoring and balanced irrigation.":
        "பயிர் ஆரோக்கியமாகத் தெரிகிறது. வழக்கமான கண்காணிப்பையும் சீரான நீர்ப்பாசனத்தையும் பராமரிக்கவும்.",
    "Early-stage symptoms detected; monitor closely and consider a preventive treatment.":
        "ஆரம்பநிலை அறிகுறிகள் கண்டறியப்பட்டன; உன்னிப்பாகக் கண்காணித்து ஒரு தடுப்பு சிகிச்சையைக் கருதவும்.",
    "Apply an appropriate fungicide/treatment and improve air circulation around affected plants.":
        "பொருத்தமான பூசணக்கொல்லி/சிகிச்சையைப் பயன்படுத்தி பாதிக்கப்பட்ட செடிகள் சுற்றிலும் "
        "காற்றோட்டத்தை மேம்படுத்தவும்.",
    "Severity is high — remove/destroy severely affected plant material and treat promptly to limit spread.":
        "தீவிரம் அதிகமாக உள்ளது — கடுமையாக பாதிக்கப்பட்ட தாவரப் பொருட்களை அகற்றவும்/அழிக்கவும், "
        "பரவலை கட்டுப்படுத்த உடனடியாக சிகிச்சை அளிக்கவும்.",
    # Closing statements by health status
    "Overall health is good — maintain current practices and keep monitoring.":
        "ஒட்டுமொத்த ஆரோக்கியம் நல்லது — தற்போதைய நடைமுறைகளை பராமரித்து தொடர்ந்து கண்காணிக்கவும்.",
    "Overall health is moderate — monitor closely and make incremental adjustments.":
        "ஒட்டுமொத்த ஆரோக்கியம் மிதமானது — உன்னிப்பாகக் கண்காணித்து படிப்படியான சரிசெய்தல்களைச் செய்யவும்.",
    "Overall health is at risk — address the largest contributing factor first and recheck within a few days.":
        "ஒட்டுமொத்த ஆரோக்கியம் ஆபத்தில் உள்ளது — மிகப்பெரிய காரணியை முதலில் சரிசெய்து சில "
        "நாட்களுக்குள் மீண்டும் சரிபார்க்கவும்.",
    "Overall health is critical — prioritize immediate intervention on both disease and environmental factors.":
        "ஒட்டுமொத்த ஆரோக்கியம் ஆபத்தானது — நோய் மற்றும் சுற்றுச்சூழல் காரணிகள் இரண்டிலும் "
        "உடனடி தலையீட்டிற்கு முன்னுரிமை அளிக்கவும்.",
}

# ---------------------------------------------------------------------------
# src/recommendation_engine.py — the rule-based engine behind Crop Health
# Analysis. Same "translate the template, keep {placeholders}" pattern.
# ---------------------------------------------------------------------------
REC_KEYWORD_LABEL_TA: dict[str, str] = {
    "blight": "கருகல்", "rust": "துரு", "mold": "பூஞ்சை", "mildew": "பூஞ்சணம்",
    "wilt": "வாடல்", "spot": "புள்ளி", "rot": "அழுகல்",
}

REC_KEYWORD_ADVICE_TA: dict[str, str] = {
    "blight": "பாதிக்கப்பட்ட இலைகளை உடனடியாக அகற்றி அழிக்கவும், மேலிருந்து நீர்ப்பாசனத்தைத் "
              "தவிர்க்கவும் — ஈரமான இலைகளில் கருகல் நோய் வேகமாகப் பரவும்.",
    "rust": "கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி, மேலாவரணத்தில் ஈரப்பதம் தேங்காதவாறு "
            "அடர்த்தியான நடவை தவிர்க்கவும்.",
    "mold": "இலைகள் சுற்றிலும் ஈரப்பதத்தைக் குறைக்க காற்றோட்டத்தை மேம்படுத்தி மேலாவரண "
            "அடர்த்தியைக் குறைக்கவும்.",
    "mildew": "காற்றோட்டத்தை மேம்படுத்தவும், நீர்ப்பாசனம் செய்யும்போது இலைகளை நனைப்பதைத் "
              "தவிர்க்கவும், தொடர்ந்து பரவினால் ஒரு தடுப்பு சிகிச்சையைப் பயன்படுத்தவும்.",
    "wilt": "வேர் ஆரோக்கியத்தையும் வடிகாலையும் சரிபார்க்கவும் — வாடல் பெரும்பாலும் ஒரு "
            "மண் வழி நோய்க்கிருமியையோ நீர் அழுத்தத்தையோ குறிக்கிறது, இலை நோயை அல்ல.",
    "spot": "பாதிக்கப்பட்ட இலைகளை அகற்றவும், இலைகள் ஈரமாக இருக்கும் நேரத்தைக் குறைக்க "
            "மேலிருந்து நீர்ப்பாசனத்தைத் தவிர்க்கவும்.",
    "rot": "வடிகாலை உடனடியாக மேம்படுத்தி பாதிக்கப்பட்ட திசுக்களை அகற்றவும் — ஈரமான "
           "மண்ணில் அழுகல் வேகமாகப் பரவும்.",
}

REC_SEVERITY_ACTION_TA: dict[str, str] = {
    "Apply an appropriate fungicide or treatment immediately, and isolate or "
    "remove severely affected plants to limit spread.":
        "பொருத்தமான பூசணக்கொல்லி அல்லது சிகிச்சையை உடனடியாகப் பயன்படுத்தவும், பரவலை "
        "கட்டுப்படுத்த கடுமையாக பாதிக்கப்பட்ட செடிகளை தனிமைப்படுத்தவும் அல்லது அகற்றவும்.",
    "Avoid working in the field while foliage is wet to prevent further spread.":
        "மேலும் பரவலைத் தடுக்க இலைகள் ஈரமாக இருக்கும்போது வயலில் வேலை செய்வதைத் தவிர்க்கவும்.",
    "Apply a targeted treatment and re-inspect the affected area every 2-3 "
    "days to confirm it's working.":
        "இலக்கு வைக்கப்பட்ட சிகிச்சையைப் பயன்படுத்தி, அது வேலை செய்கிறதா என்பதை உறுதிப்படுத்த "
        "பாதிக்கப்பட்ட பகுதியை ஒவ்வொரு 2-3 நாட்களுக்கும் மீண்டும் ஆய்வு செய்யவும்.",
    "Hold off on treatment and monitor for progression first — mild, "
    "early-stage cases often resolve with cultural controls alone.":
        "சிகிச்சையை தற்காலிகமாக நிறுத்திவைத்து முதலில் முன்னேற்றத்தைக் கண்காணிக்கவும் — லேசான, "
        "ஆரம்பநிலை வழக்குகள் பெரும்பாலும் பயிர் நிர்வாக முறைகள் மட்டும் மூலம் தீரும்.",
}

REC_ENV_ADVICE_TA: dict[str, dict[str, str]] = {
    "temperature": {
        "Low": "பயிரை குளிரிலிருந்து பாதுகாக்கவும் — வரிசை மூடிகளைப் பயன்படுத்தவும் அல்லது "
               "வெப்பநிலை-உணர்திறன் வயல் பணிகளை தாமதப்படுத்தவும்.",
        "High": "வெப்ப அழுத்தத்தைக் குறைக்க நிழல் வழங்கவும் அல்லது நாளின் குளிர்ந்த நேரங்களில் "
                "நீர்ப்பாசனம் செய்யவும்.",
        "Extreme": "வெப்பநிலை பாதுகாப்பான வரம்பிற்கு வெகு வெளியே உள்ளது — உடனடியாக பாதுகாப்பு "
                   "நடவடிக்கை எடுக்கவும் அல்லது முக்கியமான வயல் பணிகளை தாமதப்படுத்தவும்.",
    },
    "humidity": {
        "Low": "செடிகள் சுற்றிலும் ஈரப்பதத்தை அதிகரிக்கவும் (எ.கா. மல்ச் செய்தல்) அல்லது "
               "நீர்ப்பாசன நேரத்தை சரிசெய்யவும்.",
        "High": "அதிகப்படியான ஈரப்பதத்தைக் குறைத்து பூஞ்சை-நோய் அபாயத்தைக் குறைக்க "
                "காற்றோட்டத்தை மேம்படுத்தவும்.",
        "Extreme": "ஈரப்பதம் பாதுகாப்பான வரம்பிற்கு வெகு வெளியே உள்ளது — அழுத்தம் அல்லது நோய் "
                   "அறிகுறிகளுக்காக செடிகளை உன்னிப்பாகப் பரிசோதிக்கவும்.",
    },
    "soil_moisture": {
        "Low": "சீரான நீர்ப்பாசன அட்டவணையுடன் பொருத்தமான மண் ஈரப்பதத்தை பராமரிக்கவும்.",
        "High": "அதிகப்படியான நீர்ப்பாசனத்தைத் தவிர்க்கவும் — மீண்டும் நீர்ப்பாசனம் செய்யும் "
                "முன் மண் வடியட்டும்.",
        "Extreme": "வடிகால் அல்லது நீர்ப்பாசனத்தை உடனடியாக சரிசெய்யவும் — மண் ஈரப்பதம் "
                   "பாதுகாப்பான வரம்பிற்கு வெகு வெளியே உள்ளது.",
    },
    "rainfall": {
        "Low": "போதுமான நீர் விநியோகத்தை பராமரிக்க மழையை நீர்ப்பாசனத்துடன் துணைபுரியவும்.",
        "High": "நீர் தேங்குதலையும் ஊட்டச்சத்து வடிதலையும் தடுக்க போதுமான வயல் வடிகாலை "
                "உறுதி செய்யவும்.",
        "Extreme": "தாழ்வான பகுதிகளைப் பாதுகாத்து நீர் தேங்கியிருப்பதை சரிபார்க்கவும்.",
    },
}

REC_HEALTH_SCORE_TA: list[str] = [
    "ஒட்டுமொத்த ஆரோக்கியம் நல்லது — தற்போதைய நடைமுறைகளை பராமரித்து வழக்கமான "
    "கண்காணிப்பைத் தொடரவும்.",
    "ஒட்டுமொத்த ஆரோக்கியம் மிதமானது — உன்னிப்பாகக் கண்காணித்து பெரிய மாற்றங்களுக்குப் "
    "பதிலாக படிப்படியான சரிசெய்தல்களைச் செய்யவும்.",
    "ஒட்டுமொத்த ஆரோக்கியம் ஆபத்தில் உள்ளது — மிகப்பெரிய காரணியை (நோய் அல்லது "
    "சுற்றுச்சூழல்) முன்னுரிமைப்படுத்தி சில நாட்களுக்குள் மீண்டும் சரிபார்க்கவும்.",
    "ஒட்டுமொத்த ஆரோக்கியம் ஆபத்தானது — நோய் சிகிச்சை மற்றும் சுற்றுச்சூழல் திருத்தம் "
    "இரண்டிலும் உடனடியாக தலையிடவும்.",
]

REC_MISC_TA: dict[str, str] = {
    "No disease detected — keep up preventive practices such as crop "
    "rotation and clean tools to avoid introducing pathogens.":
        "நோய் எதுவும் கண்டறியப்படவில்லை — நோய்க்கிருமிகள் நுழைவதைத் தவிர்க்க பயிர் சுழற்சி "
        "மற்றும் சுத்தமான கருவிகள் போன்ற தடுப்பு நடைமுறைகளைத் தொடரவும்.",
    "Inspect affected leaves closely to confirm the extent of infection.":
        "தொற்றின் அளவை உறுதிப்படுத்த பாதிக்கப்பட்ட இலைகளை உன்னிப்பாகப் பரிசோதிக்கவும்.",
    "Monitor disease progression over the next several days to check "
    "whether treatment is working.":
        "சிகிச்சை வேலை செய்கிறதா என்பதை சரிபார்க்க அடுத்த சில நாட்களில் நோய் "
        "முன்னேற்றத்தைக் கண்காணிக்கவும்.",
    "Monitor environmental conditions closely over the next few days "
    "and re-check readings after adjustments.":
        "அடுத்த சில நாட்களில் சுற்றுச்சூழல் நிலைமைகளை உன்னிப்பாகக் கண்காணித்து, "
        "சரிசெய்தல்களுக்குப் பிறகு அளவீடுகளை மீண்டும் சரிபார்க்கவும்.",
    "All environmental conditions are within {crop}'s ideal range — maintain current practices.":
        "அனைத்து சுற்றுச்சூழல் நிலைமைகளும் {crop}-இன் சிறந்த வரம்பிற்குள் உள்ளன — "
        "தற்போதைய நடைமுறைகளை பராமரிக்கவும்.",
    "Re-check the health score input — it should be between 0 and 100.":
        "ஆரோக்கிய மதிப்பெண் உள்ளீட்டை மீண்டும் சரிபார்க்கவும் — இது 0 மற்றும் 100 "
        "இடையே இருக்க வேண்டும்.",
    "{crop} looks healthy with favorable conditions (health score {score}/100) — maintain current practices.":
        "{crop} சாதகமான நிலைமைகளுடன் ஆரோக்கியமாகத் தெரிகிறது (ஆரோக்கிய மதிப்பெண் "
        "{score}/100) — தற்போதைய நடைமுறைகளை பராமரிக்கவும்.",
    "{crop} shows no disease, but environmental conditions need attention (health score {score}/100).":
        "{crop}-இல் நோய் எதுவும் இல்லை, ஆனால் சுற்றுச்சூழல் நிலைமைகளுக்கு கவனம் "
        "தேவை (ஆரோக்கிய மதிப்பெண் {score}/100).",
    "{crop} has a detected disease but environmental conditions are favorable "
    "(health score {score}/100) — focus on treatment.":
        "{crop}-இல் ஒரு நோய் கண்டறியப்பட்டுள்ளது ஆனால் சுற்றுச்சூழல் நிலைமைகள் "
        "சாதகமாக உள்ளன (ஆரோக்கிய மதிப்பெண் {score}/100) — சிகிச்சையில் கவனம் செலுத்தவும்.",
    "{crop} needs attention on both disease and environmental fronts (health score {score}/100).":
        "{crop}-க்கு நோய் மற்றும் சுற்றுச்சூழல் ஆகிய இரண்டு அம்சங்களிலும் கவனம் தேவை "
        "(ஆரோக்கிய மதிப்பெண் {score}/100).",
    # reason-field templates (small print under each recommendation)
    "'{disease}' with severity 'None' — no disease signal present.":
        "'{disease}' தீவிரம் 'இல்லை' — நோய் அறிகுறி எதுவும் இல்லை.",
    "'{disease}' detected at {severity} severity.":
        "'{disease}' {severity} தீவிரத்தில் கண்டறியப்பட்டது.",
    "'{disease}' matches known pattern '{keyword}'.":
        "'{disease}' அறியப்பட்ட வடிவம் '{keyword}'-உடன் பொருந்துகிறது.",
    "Severity is {severity}.": "தீவிரம் {severity}.",
    "One or more readings fall outside the ideal range for this crop.":
        "இந்த பயிருக்கான சிறந்த வரம்பிற்கு வெளியே ஒன்று அல்லது அதற்கு மேற்பட்ட அளவீடுகள் உள்ளன.",
    "Temperature, humidity, soil moisture, and rainfall are all optimal.":
        "வெப்பநிலை, ஈரப்பதம், மண் ஈரப்பதம் மற்றும் மழைப்பொழிவு அனைத்தும் உகந்தவை.",
    "Health score {score} is outside the expected 0-100 range.":
        "ஆரோக்கிய மதிப்பெண் {score} எதிர்பார்க்கப்படும் 0-100 வரம்பிற்கு வெளியே உள்ளது.",
    "Temperature is {value}°C, below {crop}'s ideal minimum of {bound}°C.":
        "வெப்பநிலை {value}°C, {crop}-இன் சிறந்த குறைந்தபட்ச {bound}°C-ஐ விட குறைவு.",
    "Temperature is {value}°C, above {crop}'s ideal maximum of {bound}°C.":
        "வெப்பநிலை {value}°C, {crop}-இன் சிறந்த அதிகபட்ச {bound}°C-ஐ விட அதிகம்.",
    "Temperature is {value}°C, well outside {crop}'s safe range.":
        "வெப்பநிலை {value}°C, {crop}-இன் பாதுகாப்பான வரம்பிற்கு வெகு வெளியே.",
    "Humidity is {value}%, below {crop}'s ideal minimum of {bound}%.":
        "ஈரப்பதம் {value}%, {crop}-இன் சிறந்த குறைந்தபட்ச {bound}%-ஐ விட குறைவு.",
    "Humidity is {value}%, above {crop}'s ideal maximum of {bound}%.":
        "ஈரப்பதம் {value}%, {crop}-இன் சிறந்த அதிகபட்ச {bound}%-ஐ விட அதிகம்.",
    "Humidity is {value}%, well outside {crop}'s safe range.":
        "ஈரப்பதம் {value}%, {crop}-இன் பாதுகாப்பான வரம்பிற்கு வெகு வெளியே.",
    "Soil moisture is {value}%, below {crop}'s ideal minimum of {bound}%.":
        "மண் ஈரப்பதம் {value}%, {crop}-இன் சிறந்த குறைந்தபட்ச {bound}%-ஐ விட குறைவு.",
    "Soil moisture is {value}%, above {crop}'s ideal maximum of {bound}%.":
        "மண் ஈரப்பதம் {value}%, {crop}-இன் சிறந்த அதிகபட்ச {bound}%-ஐ விட அதிகம்.",
    "Soil moisture is {value}%, well outside {crop}'s safe range.":
        "மண் ஈரப்பதம் {value}%, {crop}-இன் பாதுகாப்பான வரம்பிற்கு வெகு வெளியே.",
    "Rainfall is {value}mm, below {crop}'s ideal minimum of {bound}mm.":
        "மழைப்பொழிவு {value}mm, {crop}-இன் சிறந்த குறைந்தபட்ச {bound}mm-ஐ விட குறைவு.",
    "Rainfall is {value}mm, above {crop}'s ideal maximum of {bound}mm.":
        "மழைப்பொழிவு {value}mm, {crop}-இன் சிறந்த அதிகபட்ச {bound}mm-ஐ விட அதிகம்.",
    "Rainfall is {value}mm, well outside {crop}'s safe range.":
        "மழைப்பொழிவு {value}mm, {crop}-இன் பாதுகாப்பான வரம்பிற்கு வெகு வெளியே.",
    "Health score is {score}/100 (Healthy band, 80-100).":
        "ஆரோக்கிய மதிப்பெண் {score}/100 (ஆரோக்கியமான வரம்பு, 80-100).",
    "Health score is {score}/100 (Moderate band, 60-79).":
        "ஆரோக்கிய மதிப்பெண் {score}/100 (மிதமான வரம்பு, 60-79).",
    "Health score is {score}/100 (At Risk band, 40-59).":
        "ஆரோக்கிய மதிப்பெண் {score}/100 (ஆபத்தான வரம்பு, 40-59).",
    "Health score is {score}/100 (Critical band, 0-39).":
        "ஆரோக்கிய மதிப்பெண் {score}/100 (ஆபத்தான வரம்பு, 0-39).",
}

HEALTH_PAGE_TA: dict[str, str] = {
    "Crop Health Analysis": "பயிர் ஆரோக்கிய பகுப்பாய்வு",
    "Combine disease detection and environmental data into an overall health score.":
        "நோய் கண்டறிதலையும் சுற்றுச்சூழல் தரவையும் ஒரே ஒட்டுமொத்த ஆரோக்கிய மதிப்பெண்ணாக இணைக்கவும்.",
    "No trained disease model found.": "பயிற்சி பெற்ற நோய் மாதிரி எதுவும் இல்லை.",
    "Train one first from the Disease Detection page's instructions before "
    "running a health analysis.":
        "ஆரோக்கிய பகுப்பாய்வை இயக்கும் முன் நோய் கண்டறிதல் பக்கத்தின் வழிமுறைகளில் "
        "இருந்து முதலில் ஒன்றைப் பயிற்சி செய்யவும்.",
    "Inputs": "உள்ளீடுகள்",
    "Disease result": "நோய் முடிவு",
    "Crop name": "பயிர் பெயர்",
    "Detected disease": "கண்டறியப்பட்ட நோய்",
    "Disease confidence": "நோய் நம்பகத்தன்மை",
    "Use the plus and minus buttons to change confidence by 1%.":
        "நம்பகத்தன்மையை 1% மாற்ற கூட்டல் மற்றும் கழித்தல் பொத்தான்களைப் பயன்படுத்தவும்.",
    "Disease severity": "நோய் தீவிரம்",
    "Environmental readings": "சுற்றுச்சூழல் அளவீடுகள்",
    "Calculate crop health": "பயிர் ஆரோக்கியத்தைக் கணக்கிடவும்",
    "Analysis result": "பகுப்பாய்வு முடிவு",
    "Environmental risk model not found.": "சுற்றுச்சூழல் ஆபத்து மாதிரி கிடைக்கவில்லை.",
    "Calculating crop health failed unexpectedly. Please try "
    "again. If the problem continues, contact the app maintainer.":
        "பயிர் ஆரோக்கியத்தைக் கணக்கிடுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Awaiting calculation": "கணக்கீட்டிற்காக காத்திருக்கிறது",
    "Click **Calculate crop health** to combine the disease result "
    "and environmental readings into an overall score and status.":
        "நோய் முடிவையும் சுற்றுச்சூழல் அளவீடுகளையும் ஒரே ஒட்டுமொத்த மதிப்பெண் மற்றும் "
        "நிலையாக இணைக்க **பயிர் ஆரோக்கியத்தைக் கணக்கிடவும்** என்பதைக் கிளிக் செய்யவும்.",
    "Overall crop status": "ஒட்டுமொத்த பயிர் நிலை",
    "Crop:": "பயிர்:",
    "Environmental risk model:": "சுற்றுச்சூழல் ஆபத்து மாதிரி:",
    "Risk breakdown": "ஆபத்து பிரிவு",
    "Disease risk": "நோய் ஆபத்து",
    "Disease score:": "நோய் மதிப்பெண்:",
    "Environmental risk": "சுற்றுச்சூழல் ஆபத்து",
    "Env. score:": "சுற்றுச்சூழல் மதிப்பெண்:",
    "Detailed metrics": "விரிவான அளவீடுகள்",
    "Disease": "நோய்",
    "Environment": "சுற்றுச்சூழல்",
    "Why this score?": "ஏன் இந்த மதிப்பெண்?",
    "Agricultural recommendation": "விவசாய பரிந்துரை",
    "Priority actions": "முன்னுரிமை நடவடிக்கைகள்",
    "Analysis saved to database (ID:": "பகுப்பாய்வு தரவுத்தளத்தில் சேமிக்கப்பட்டது (ID:",
    "Saved": "சேமிக்கப்பட்டது",
    "Save Analysis": "பகுப்பாய்வை சேமிக்கவும்",
    # category/priority badge words
    "disease": "நோய்",
    "environment": "சுற்றுச்சூழல்",
    "overall": "ஒட்டுமொத்தம்",
    "HIGH": "அதிகம்",
    "MEDIUM": "நடுத்தரம்",
    "LOW": "குறைவு",
}

# ---------------------------------------------------------------------------
# Home page (app.py)
# ---------------------------------------------------------------------------
HOME_TA: dict[str, str] = {
    "View Outbreak Alerts →": "பரவல் எச்சரிக்கைகளைப் பார்க்கவும் →",
    "**Multi-crop health workflow** · pick a crop, upload a leaf image, "
    "review the trained disease prediction, enter environmental readings, "
    "and calculate an explainable health score.":
        "**பல பயிர் ஆரோக்கிய பணிப்பாய்வு** · ஒரு பயிரைத் தேர்ந்தெடுக்கவும், இலைப் "
        "படத்தைப் பதிவேற்றவும், பயிற்சி பெற்ற நோய் முன்னறிவிப்பை பரிசீலிக்கவும், "
        "சுற்றுச்சூழல் அளவீடுகளை உள்ளிடவும், விளக்கமளிக்கக்கூடிய ஆரோக்கிய மதிப்பெண்ணைக் கணக்கிடவும்.",
    "Monitor crop health and detect plant diseases **early** using AI-driven "
    "image analysis combined with environmental data — entirely in software, "
    "with no sensors or cameras required.":
        "AI-இயங்கும் பட பகுப்பாய்வையும் சுற்றுச்சூழல் தரவையும் இணைத்து, எந்த "
        "சென்சார்களும் கேமராக்களும் இல்லாமல், முழுவதுமாக மென்பொருள் மூலம் பயிர் "
        "ஆரோக்கியத்தை கண்காணித்து தாவர நோய்களை **முன்கூட்டியே** கண்டறியவும்.",
    "What you can do here": "இங்கே நீங்கள் என்ன செய்யலாம்",
    "Where to start": "எங்கிருந்து தொடங்குவது",
    "Use the **sidebar on the left** to navigate between modules. A suggested "
    "flow: **Disease Detection → Environmental Analysis → Crop Health "
    "Analysis → Dashboard**.":
        "பிரிவுகளுக்கு இடையே செல்ல **இடதுபுறம் உள்ள பக்கப்பட்டியை** பயன்படுத்தவும். "
        "பரிந்துரைக்கப்படும் வழிமுறை: **நோய் கண்டறிதல் → சுற்றுச்சூழல் பகுப்பாய்வு → "
        "பயிர் ஆரோக்கிய பகுப்பாய்வு → கட்டுப்பாட்டு பலகை**.",
    "Supported crops": "ஆதரிக்கப்படும் பயிர்கள்",
    "No trained disease models were found yet. Train one with "
    "`src/model_training.py --crop <name>` to see it listed here.":
        "இதுவரை பயிற்சி பெற்ற நோய் மாதிரிகள் எதுவும் இல்லை. அதை இங்கே பட்டியலிட "
        "`src/model_training.py --crop <name>` மூலம் ஒன்றைப் பயிற்சி செய்யவும்.",
    "Your activity": "உங்கள் செயல்பாடு",
    "Health analyses": "ஆரோக்கிய பகுப்பாய்வுகள்",
    "Disease detections": "நோய் கண்டறிதல்கள்",
    "Environmental readings": "சுற்றுச்சூழல் அளவீடுகள்",
    "Loading your activity stats failed unexpectedly. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "உங்கள் செயல்பாட்டு புள்ளிவிவரங்களை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. "
        "மீண்டும் முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Detect diseases": "நோய்களைக் கண்டறியவும்",
    "Upload a leaf image for any supported crop; the trained model reports "
    "disease, confidence, and severity.":
        "ஆதரிக்கப்படும் எந்த பயிரின் இலைப் படத்தையும் பதிவேற்றவும்; பயிற்சி பெற்ற "
        "மாதிரி நோய், நம்பகத்தன்மை மற்றும் தீவிரத்தை தெரிவிக்கும்.",
    "Scan a field": "வயலை ஆய்வு செய்யவும்",
    "Upload 10-20+ leaf photos from a field walk and get one aggregated "
    "field health report.":
        "வயல் சுற்றுப்பயணத்திலிருந்து 10-20+ இலைப் புகைப்படங்களைப் பதிவேற்றி ஒரு "
        "ஒருங்கிணைந்த வயல் ஆரோக்கிய அறிக்கையைப் பெறவும்.",
    "Watch for outbreaks": "பரவல்களைக் கவனிக்கவும்",
    "See which crops are trending worse across your saved analysis history.":
        "உங்கள் சேமிக்கப்பட்ட பகுப்பாய்வு வரலாற்றில் எந்த பயிர்கள் மோசமடைந்து "
        "வருகின்றன என்பதைப் பார்க்கவும்.",
    "Track conditions": "நிலைமைகளைக் கண்காணிக்கவும்",
    "Log temperature, humidity, soil moisture and rainfall.":
        "வெப்பநிலை, ஈரப்பதம், மண் ஈரப்பதம் மற்றும் மழைப்பொழிவை பதிவு செய்யவும்.",
    "Score crop health": "பயிர் ஆரோக்கியத்தை மதிப்பிடவும்",
    "Combine the image and environmental signals into one health score.":
        "படம் மற்றும் சுற்றுச்சூழல் தரவுகளை ஒரே ஆரோக்கிய மதிப்பெண்ணாக இணைக்கவும்.",
    "Visualize trends": "போக்குகளைக் காட்சிப்படுத்தவும்",
    "See stats and charts across all your past analyses.":
        "உங்கள் அனைத்து கடந்த பகுப்பாய்வுகளின் புள்ளிவிவரங்கள் மற்றும் விளக்கப்படங்களைப் பார்க்கவும்.",
    "Keep history": "வரலாற்றை வைத்திருக்கவும்",
    "Save completed analyses and review them later.":
        "முடிக்கப்பட்ட பகுப்பாய்வுகளை சேமித்து பின்னர் மதிப்பாய்வு செய்யவும்.",
}


# ---------------------------------------------------------------------------
# About Project page (pages/about.py)
# ---------------------------------------------------------------------------
ABOUT_TA: dict[str, str] = {
    "**Smart Crop Health Monitoring** is a software-only Streamlit "
    "application for checking crop health. It combines trained "
    "image classification, environmental risk analysis, an explainable "
    "health score, and practical agricultural recommendations in one "
    "workflow. No sensors or IoT hardware are required.":
        "**ஸ்மார்ட் பயிர் ஆரோக்கிய கண்காணிப்பு** என்பது பயிர் ஆரோக்கியத்தை "
        "சரிபார்க்கும் ஒரு மென்பொருள் மட்டும் கொண்ட Streamlit செயலி ஆகும். இது "
        "பயிற்சி பெற்ற பட வகைப்பாடு, சுற்றுச்சூழல் ஆபத்து பகுப்பாய்வு, விளக்கமளிக்கக்கூடிய "
        "ஆரோக்கிய மதிப்பெண் மற்றும் நடைமுறை விவசாய பரிந்துரைகளை ஒரே பணிப்பாய்வில் "
        "இணைக்கிறது. எந்த சென்சார்களும் IoT வன்பொருளும் தேவையில்லை.",
    "Current workflow": "தற்போதைய பணிப்பாய்வு",
    "Pick a crop and upload a leaf image": "பயிரைத் தேர்ந்தெடுத்து இலைப் படத்தைப் பதிவேற்றவும்",
    "Choose a trained crop and submit a clear JPG or PNG image for analysis.":
        "பயிற்சி பெற்ற பயிரைத் தேர்ந்தெடுத்து, பகுப்பாய்வுக்காக தெளிவான JPG அல்லது PNG படத்தைச் சமர்ப்பிக்கவும்.",
    "Classify the leaf": "இலையை வகைப்படுத்தவும்",
    "That crop's trained model recognizes Healthy, Early Blight, or Late Blight.":
        "அந்த பயிரின் பயிற்சி பெற்ற மாதிரி ஆரோக்கியமானது, ஆரம்பநிலை கருகல் அல்லது தாமதமான கருகல் ஆகியவற்றை அடையாளம் காணும்.",
    "Review confidence": "நம்பகத்தன்மையை மதிப்பாய்வு செய்யவும்",
    "The prediction includes class probabilities and confidence.":
        "முன்னறிவிப்பில் வகை நிகழ்தகவுகளும் நம்பகத்தன்மையும் அடங்கும்.",
    "Enter environmental readings": "சுற்றுச்சூழல் அளவீடுகளை உள்ளிடவும்",
    "Provide temperature, humidity, soil moisture, and rainfall.":
        "வெப்பநிலை, ஈரப்பதம், மண் ஈரப்பதம் மற்றும் மழைப்பொழிவை வழங்கவும்.",
    "Assess environmental risk": "சுற்றுச்சூழல் ஆபத்தை மதிப்பிடவும்",
    "The trained environmental model classifies the current conditions.":
        "பயிற்சி பெற்ற சுற்றுச்சூழல் மாதிரி தற்போதைய நிலைமைகளை வகைப்படுத்துகிறது.",
    "Calculate crop health": "பயிர் ஆரோக்கியத்தைக் கணக்கிடவும்",
    "Disease and environmental signals become one explainable score out of 100.":
        "நோய் மற்றும் சுற்றுச்சூழல் சமிக்ஞைகள் 100-க்கு ஒரு விளக்கமளிக்கக்கூடிய மதிப்பெண்ணாக மாறும்.",
    "Review recommendations": "பரிந்துரைகளை மதிப்பாய்வு செய்யவும்",
    "Severity-aware, crop-specific actions explain what to do next.":
        "தீவிரத்தை உணர்ந்த, பயிருக்கு உரிய நடவடிக்கைகள் அடுத்து என்ன செய்ய வேண்டும் என்பதை விளக்குகின்றன.",
    "Save and review analyses": "பகுப்பாய்வுகளை சேமித்து மதிப்பாய்வு செய்யவும்",
    "Save completed assessments to SQLite and inspect them in History and Dashboard.":
        "முடிக்கப்பட்ட மதிப்பீடுகளை SQLite-ல் சேமித்து வரலாறு மற்றும் கட்டுப்பாட்டு பலகையில் பரிசோதிக்கவும்.",
    "Technology": "தொழில்நுட்பம்",
    "Frontend / UI": "முன் பகுதி / UI",
    "Programming": "நிரலாக்கம்",
    "AI / Deep Learning": "AI / ஆழ்ந்த கற்றல்",
    "Machine Learning": "இயந்திர கற்றல்",
    "Image Processing": "பட செயலாக்கம்",
    "Data Processing": "தரவு செயலாக்கம்",
    "Database": "தரவுத்தளம்",
    "Visualization": "காட்சிப்படுத்தல்",
    "Current model scope": "தற்போதைய மாதிரி வரம்பு",
    "No trained disease model found yet.": "இதுவரை பயிற்சி பெற்ற நோய் மாதிரி எதுவும் இல்லை.",
    "Currently trained:": "தற்போது பயிற்சி பெற்றவை:",
    "none": "எதுவுமில்லை",
    "can be added by training a model for that crop — "
    "see the Disease Detection page for the training command.":
        "அந்த பயிருக்கான மாதிரியைப் பயிற்சி செய்வதன் மூலம் சேர்க்கலாம் — "
        "பயிற்சிக் கட்டளைக்கு நோய் கண்டறிதல் பக்கத்தைப் பார்க்கவும்.",
}

# ---------------------------------------------------------------------------
# Outbreak Alerts page (pages/alerts.py) + its dynamic risk_reason sentences
# (src/outbreak_detection.py, translated via tr_template like health_engine.py)
# ---------------------------------------------------------------------------
ALERTS_TA: dict[str, str] = {
    "Outbreak Alerts": "பரவல் எச்சரிக்கைகள்",
    "Rolling-window trend detection over your saved Disease Detection and Field Scan history.":
        "உங்கள் சேமிக்கப்பட்ட நோய் கண்டறிதல் மற்றும் வயல் ஆய்வு வரலாற்றின் மீதான உருள் சாளர போக்கு கண்டறிதல்.",
    "Rolling window (saved analyses)": "உருள் சாளரம் (சேமிக்கப்பட்ட பகுப்பாய்வுகள்)",
    "Each crop's most recent N saved analyses are compared against the N before them.":
        "ஒவ்வொரு பயிரின் மிக சமீபத்திய N சேமிக்கப்பட்ட பகுப்பாய்வுகள் அதற்கு முந்தைய N உடன் ஒப்பிடப்படுகின்றன.",
    "Analyzing saved history…": "சேமிக்கப்பட்ட வரலாறு பகுப்பாய்வு செய்யப்படுகிறது…",
    "Couldn't analyze saved history right now. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "இப்போது சேமிக்கப்பட்ட வரலாற்றை பகுப்பாய்வு செய்ய முடியவில்லை. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "No history yet": "இதுவரை வரலாறு இல்லை",
    "Save a few Disease Detection or Field Scan analyses first — "
    "Outbreak Alerts needs some saved history per crop before it "
    "can compare a recent window against a prior one.":
        "முதலில் சில நோய் கண்டறிதல் அல்லது வயல் ஆய்வு பகுப்பாய்வுகளை சேமிக்கவும் — "
        "பரவல் எச்சரிக்கைகள், சமீபத்திய சாளரத்தை முந்தைய சாளரத்துடன் ஒப்பிடுவதற்கு முன், "
        "ஒவ்வொரு பயிருக்கும் சில சேமிக்கப்பட்ட வரலாறு தேவைப்படுகிறது.",
    "Risk by crop": "பயிர்வாரியான ஆபத்து",
    "No crops are currently trending worse — everything with enough "
    "history is Watch level or better.":
        "தற்போது எந்த பயிரும் மோசமடையவில்லை — போதிய வரலாறு உள்ள அனைத்தும் "
        "'கவனி' நிலை அல்லது அதற்கு மேல் உள்ளன.",
    "crop(s) trending worse:": "பயிர்(கள்) மோசமடைந்து வருகின்றன:",
    "Recent window": "சமீபத்திய சாளரம்",
    "Prior window": "முந்தைய சாளரம்",
    "saved": "சேமிக்கப்பட்டது",
    "leaves": "இலைகள்",
    "diseased": "நோய்வாய்ப்பட்டவை",
    "high severity": "அதிக தீவிரம்",
    "Dominant:": "முதன்மை:",
    "Not enough history yet": "இதுவரை போதிய வரலாறு இல்லை",
    "Change vs. prior window": "முந்தைய சாளரத்துடன் ஒப்பிடும்போது மாற்றம்",
    "Diseased": "நோய்வாய்ப்பட்டவை",
    "High severity": "அதிக தீவிரம்",
    "pts": "புள்ளிகள்",
    "N/A": "பொருந்தாது",
    "Saved analyses (recent window)": "சேமிக்கப்பட்ட பகுப்பாய்வுகள் (சமீபத்திய சாளரம்)",
}

OUTBREAK_TA: dict[str, str] = {
    "Only {n} saved analysis(es) so far for this crop — "
    "a few more are needed before a trend can be judged.":
        "இந்த பயிருக்கு இதுவரை {n} சேமிக்கப்பட்ட பகுப்பாய்வு(கள்) மட்டுமே உள்ளன — "
        "ஒரு போக்கை மதிப்பிடுவதற்கு முன் இன்னும் சில தேவை.",
    "{pct:.0f}% of recent analyses are diseased. No prior window yet to compare against.":
        "சமீபத்திய பகுப்பாய்வுகளில் {pct:.0f}% நோய்வாய்ப்பட்டவை. ஒப்பிட முந்தைய சாளரம் இன்னும் இல்லை.",
    "No prior window yet to compare against; current detections look manageable.":
        "ஒப்பிட முந்தைய சாளரம் இன்னும் இல்லை; தற்போதைய கண்டறிதல்கள் கையாளக்கூடியதாகத் தெரிகிறது.",
    "{pct:.0f}% of recent analyses are diseased, and high-severity share is up "
    "{delta:+.0f} pts vs. the prior window.":
        "சமீபத்திய பகுப்பாய்வுகளில் {pct:.0f}% நோய்வாய்ப்பட்டவை, மேலும் அதிக-தீவிர பங்கு "
        "முந்தைய சாளரத்தை விட {delta:+.0f} புள்ளிகள் அதிகரித்துள்ளது.",
    "{pct:.0f}% of recent analyses are diseased.":
        "சமீபத்திய பகுப்பாய்வுகளில் {pct:.0f}% நோய்வாய்ப்பட்டவை.",
    "Diseased share is up {delta:+.0f} pts vs. the prior window, "
    "high-severity share up {hdelta:+.0f} pts.":
        "நோய்வாய்ப்பட்ட பங்கு முந்தைய சாளரத்தை விட {delta:+.0f} புள்ளிகள் அதிகரித்துள்ளது, "
        "அதிக-தீவிர பங்கு {hdelta:+.0f} புள்ளிகள் அதிகரித்துள்ளது.",
    "Diseased share is up {delta:+.0f} pts vs. the prior window.":
        "நோய்வாய்ப்பட்ட பங்கு முந்தைய சாளரத்தை விட {delta:+.0f} புள்ளிகள் அதிகரித்துள்ளது.",
    "Diseased share is {pct:.0f}% and trending up {delta:+.0f} pts.":
        "நோய்வாய்ப்பட்ட பங்கு {pct:.0f}% ஆக உள்ளது மற்றும் {delta:+.0f} புள்ளிகள் அதிகரித்து வருகிறது.",
    "Diseased share is {pct:.0f}%, stable or improving ({delta_txt}).":
        "நோய்வாய்ப்பட்ட பங்கு {pct:.0f}% ஆக உள்ளது, நிலையானது அல்லது மேம்படுகிறது ({delta_txt}).",
    "stable": "நிலையானது",
}

# ---------------------------------------------------------------------------
# Analysis History page (pages/history.py)
# ---------------------------------------------------------------------------
HISTORY_TA: dict[str, str] = {
    "Analysis History": "பகுப்பாய்வு வரலாறு",
    "Review, filter, and manage previously saved analyses.":
        "முன்பு சேமிக்கப்பட்ட பகுப்பாய்வுகளை மதிப்பாய்வு செய்து, வடிகட்டி, நிர்வகிக்கவும்.",
    "Crop Health": "பயிர் ஆரோக்கியம்",
    "Disease Detection": "நோய் கண்டறிதல்",
    "Environmental": "சுற்றுச்சூழல்",
    "Filters & sorting": "வடிகட்டிகள் & வரிசைப்படுத்துதல்",
    "Crop": "பயிர்",
    "Disease": "நோய்",
    "Risk level": "ஆபத்து அளவு",
    "Date range": "தேதி வரம்பு",
    "Sort by": "வரிசைப்படுத்து",
    "All": "அனைத்தும்",
    "Date (newest first)": "தேதி (புதியது முதலில்)",
    "Date (oldest first)": "தேதி (பழையது முதலில்)",
    "Health score (high to low)": "ஆரோக்கிய மதிப்பெண் (அதிகம் முதல் குறைவு)",
    "Health score (low to high)": "ஆரோக்கிய மதிப்பெண் (குறைவு முதல் அதிகம்)",
    "Confidence (high to low)": "நம்பகத்தன்மை (அதிகம் முதல் குறைவு)",
    "Confidence (low to high)": "நம்பகத்தன்மை (குறைவு முதல் அதிகம்)",
    "Crop (A-Z)": "பயிர் (A-Z)",
    "analyses": "பகுப்பாய்வுகள்",
    "No analyses match the current filters.": "தற்போதைய வடிகட்டிகளுடன் பொருந்தும் பகுப்பாய்வுகள் இல்லை.",
    "Records": "பதிவுகள்",
    "Expand a record to view full details or delete it.":
        "முழு விவரங்களைப் பார்க்க அல்லது நீக்க ஒரு பதிவை விரிவாக்கவும்.",
    "Expand a record to view the analyzed image and full details, or delete it.":
        "பகுப்பாய்வு செய்யப்பட்ட படம் மற்றும் முழு விவரங்களைப் பார்க்க அல்லது நீக்க ஒரு பதிவை விரிவாக்கவும்.",
    "ID": "ID",
    "Date": "தேதி",
    "Confidence": "நம்பகத்தன்மை",
    "Severity": "தீவிரம்",
    "Health score": "ஆரோக்கிய மதிப்பெண்",
    "Risk": "ஆபத்து",
    "Recommendation": "பரிந்துரை",
    "Result": "முடிவு",
    "Temp (°C)": "வெப்பநிலை (°C)",
    "Humidity (%)": "ஈரப்பதம் (%)",
    "Soil moist. (%)": "மண் ஈரப்பதம் (%)",
    "Rainfall (mm)": "மழைப்பொழிவு (mm)",
    "Status": "நிலை",
    "Model confidence": "மாதிரி நம்பகத்தன்மை",
    "Disease risk": "நோய் ஆபத்து",
    "Environmental risk": "சுற்றுச்சூழல் ஆபத்து",
    "Temperature": "வெப்பநிலை",
    "Soil moisture": "மண் ஈரப்பதம்",
    "Rainfall": "மழைப்பொழிவு",
    "No crop health analyses saved yet. Go to "
    "<b>Crop Health Analysis</b>, run a calculation, and click "
    "<b>Save Analysis</b> to see records here.":
        "இதுவரை பயிர் ஆரோக்கிய பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>பயிர் ஆரோக்கிய "
        "பகுப்பாய்வு</b> பக்கத்திற்குச் சென்று, ஒரு கணக்கீட்டை இயக்கி, பதிவுகளைக் காண "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "No disease detection analyses saved yet. Go to "
    "<b>Disease Detection</b>, analyze a leaf image, and click "
    "<b>Save Analysis</b> to see records here.":
        "இதுவரை நோய் கண்டறிதல் பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>நோய் கண்டறிதல்</b> "
        "பக்கத்திற்குச் சென்று, ஒரு இலைப் படத்தை பகுப்பாய்வு செய்து, பதிவுகளைக் காண "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "No environmental analyses saved yet. Go to "
    "<b>Environmental Analysis</b>, assess a reading, and click "
    "<b>Save Analysis</b> to see records here.":
        "இதுவரை சுற்றுச்சூழல் பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>சுற்றுச்சூழல் "
        "பகுப்பாய்வு</b> பக்கத்திற்குச் சென்று, ஒரு அளவீட்டை மதிப்பிட்டு, பதிவுகளைக் காண "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "No recommendation recorded.": "பரிந்துரை எதுவும் பதிவு செய்யப்படவில்லை.",
    "Image not available (file may have been moved or removed).":
        "படம் கிடைக்கவில்லை (கோப்பு நகர்த்தப்பட்டிருக்கலாம் அல்லது அகற்றப்பட்டிருக்கலாம்).",
    "Analyzed leaf": "பகுப்பாய்வு செய்யப்பட்ட இலை",
    "Delete {label}": "{label} நீக்கவும்",
    "Delete this {label} (#{row_id})? This cannot be undone.":
        "இந்த {label}-ஐ (#{row_id}) நீக்கவா? இதை மீட்டெடுக்க முடியாது.",
    "Yes, delete": "ஆம், நீக்கவும்",
    "Cancel": "ரத்துசெய்",
    "{label} #{row_id} deleted.": "{label} #{row_id} நீக்கப்பட்டது.",
    "crop health analysis": "பயிர் ஆரோக்கிய பகுப்பாய்வு",
    "disease analysis": "நோய் பகுப்பாய்வு",
    "environmental analysis": "சுற்றுச்சூழல் பகுப்பாய்வு",
    "Loading {label} history failed unexpectedly. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "{label} வரலாற்றை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
}

# ---------------------------------------------------------------------------
# Field Scan page (pages/field_scan.py) — remaining UI strings beyond the
# disease/crop names already covered by tr_disease/tr_crop.
# ---------------------------------------------------------------------------
FIELD_SCAN_TA: dict[str, str] = {
    "Field Scan": "வயல் ஆய்வு",
    "Upload a batch of leaf photos from a field walk and get one aggregated health report.":
        "வயல் சுற்றுப்பயணத்திலிருந்து இலைப் புகைப்படங்களின் தொகுப்பைப் பதிவேற்றி ஒரு "
        "ஒருங்கிணைந்த ஆரோக்கிய அறிக்கையைப் பெறவும்.",
    "No trained model found.": "பயிற்சி பெற்ற மாதிரி எதுவும் இல்லை.",
    "Train a disease model first — see the Disease Detection page for instructions.":
        "முதலில் ஒரு நோய் மாதிரியைப் பயிற்சி செய்யவும் — வழிமுறைகளுக்கு நோய் கண்டறிதல் பக்கத்தைப் பார்க்கவும்.",
    "Model unavailable.": "மாதிரி கிடைக்கவில்லை.",
    "{crop}'s model file couldn't be loaded even though it's listed as "
    "trained — check the server logs for details.":
        "{crop} மாதிரிக் கோப்பு பயிற்சி பெற்றதாக பட்டியலிடப்பட்டிருந்தும் ஏற்ற முடியவில்லை — "
        "விவரங்களுக்கு சேவையக பதிவுகளைச் சரிபார்க்கவும்.",
    "1 · Upload leaf photos": "1 · இலைப் புகைப்படங்களைப் பதிவேற்றவும்",
    "Leaf images (JPG / PNG) — up to {n} at once":
        "இலைப் படங்கள் (JPG / PNG) — ஒரே நேரத்தில் {n} வரை",
    "photo(s) uploaded — only the first {n} "
    "will be scanned. Split larger batches into multiple scans.":
        " புகைப்படங்கள் பதிவேற்றப்பட்டன — முதல் {n} மட்டுமே "
        "ஆய்வு செய்யப்படும். பெரிய தொகுப்புகளை பல ஆய்வுகளாகப் பிரிக்கவும்.",
    "photo(s) ready to scan.": "புகைப்படங்கள் ஆய்வுக்குத் தயார்.",
    "Drop 10-20+ leaf photos here — one field walk, one report.":
        "இங்கே 10-20+ இலைப் புகைப்படங்களை விடவும் — ஒரு வயல் சுற்றுப்பயணம், ஒரு அறிக்கை.",
    "Advanced options": "மேம்பட்ட விருப்பங்கள்",
    "Confidence threshold": "நம்பகத்தன்மை வரம்பு",
    "Per-leaf predictions below this confidence are flagged as uncertain.":
        "இந்த நம்பகத்தன்மைக்குக் கீழ் உள்ள ஒவ்வொரு இலை முன்னறிவிப்புகளும் உறுதியற்றதாகக் குறிக்கப்படும்.",
    "Preprocessing (applied to every photo in the batch)":
        "முன்செயலாக்கம் (தொகுப்பில் உள்ள ஒவ்வொரு புகைப்படத்திற்கும் பயன்படுத்தப்படும்)",
    "Noise reduction": "இரைச்சல் குறைப்பு",
    "Background handling": "பின்னணி கையாளுதல்",
    "Run Field Scan": "வயல் ஆய்வை இயக்கவும்",
    "2 · Field health report": "2 · வயல் ஆரோக்கிய அறிக்கை",
    "The field scan failed unexpectedly. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "வயல் ஆய்வு எதிர்பாராத முறையில் தோல்வியடைந்தது. மீண்டும் முயற்சிக்கவும். "
        "சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Awaiting scan": "ஆய்வுக்காகக் காத்திருக்கிறது",
    "Upload several leaf photos and click **Run Field Scan** to see the "
    "aggregated field health report — % healthy vs diseased, dominant "
    "disease, severity breakdown, and a field health score.":
        "பல இலைப் புகைப்படங்களைப் பதிவேற்றி, ஒருங்கிணைந்த வயல் ஆரோக்கிய அறிக்கையைப் "
        "பார்க்க **வயல் ஆய்வை இயக்கவும்** என்பதைக் கிளிக் செய்யவும் — % ஆரோக்கியம் "
        "எதிராக நோய்வாய்ப்பட்டவை, முதன்மை நோய், தீவிர பிரிவு, மற்றும் வயல் ஆரோக்கிய மதிப்பெண்.",
    "None of the uploaded photos could be analyzed. See the issues below.":
        "பதிவேற்றப்பட்ட புகைப்படங்கள் எதுவும் பகுப்பாய்வு செய்ய முடியவில்லை. கீழே உள்ள சிக்கல்களைப் பார்க்கவும்.",
    "Photos scanned": "ஆய்வு செய்யப்பட்ட புகைப்படங்கள்",
    "Healthy": "ஆரோக்கியமானது",
    "leaves": "இலைகள்",
    "Dominant disease": "முதன்மை நோய்",
    "None detected": "எதுவும் கண்டறியப்படவில்லை",
    "Diseased leaves": "நோய்வாய்ப்பட்ட இலைகள்",
    "Field health score": "வயல் ஆரோக்கிய மதிப்பெண்",
    "Disease breakdown across the field": "வயல் முழுவதும் நோய் பிரிவு",
    "Leaves": "இலைகள்",
    "Severity breakdown": "தீவிர பிரிவு",
    "Individual leaves": "தனிப்பட்ட இலைகள்",
    "low confidence": "குறைந்த நம்பகத்தன்மை",
    "confidence": "நம்பகத்தன்மை",
    "photo(s) skipped": "புகைப்படங்கள் தவிர்க்கப்பட்டன",
    "New Scan": "புதிய ஆய்வு",
    "Download PDF Report": "PDF அறிக்கையைப் பதிவிறக்கவும்",
    "Field scan saved to database (ID: {id}).":
        "வயல் ஆய்வு தரவுத்தளத்தில் சேமிக்கப்பட்டது (ID: {id}).",
    "Saved": "சேமிக்கப்பட்டது",
    "Save Field Scan": "வயல் ஆய்வை சேமிக்கவும்",
    "Saving field scan…": "வயல் ஆய்வு சேமிக்கப்படுகிறது…",
    "Couldn't generate the PDF report right now. Please try again.":
        "இப்போது PDF அறிக்கையை உருவாக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",
}

# ---------------------------------------------------------------------------
# Disease Detection page (pages/disease.py)
# ---------------------------------------------------------------------------
DISEASE_PAGE_TA: dict[str, str] = {
    "Disease Detection": "நோய் கண்டறிதல்",
    "Upload a crop leaf image to detect diseases with AI.":
        "AI மூலம் நோய்களைக் கண்டறிய ஒரு பயிர் இலைப் படத்தைப் பதிவேற்றவும்.",
    "No trained model found.": "பயிற்சி பெற்ற மாதிரி எதுவும் இல்லை.",
    "Train one first using": "முதலில் இதைப் பயன்படுத்தி ஒன்றைப் பயிற்சி செய்யவும்",
    "(swap": "(மாற்றவும்",
    "for any crop in": "எந்த பயிருக்கும்",
    "If a model file exists but still won't load, check the server logs for details.":
        "ஒரு மாதிரிக் கோப்பு இருந்தும் ஏற்றப்படவில்லை என்றால், விவரங்களுக்கு சேவையக பதிவுகளைச் சரிபார்க்கவும்.",
    "Model unavailable.": "மாதிரி கிடைக்கவில்லை.",
    "{crop}'s model file couldn't be loaded even though it's listed as "
    "trained — check the server logs for details.":
        "{crop} மாதிரிக் கோப்பு பயிற்சி பெற்றதாக பட்டியலிடப்பட்டிருந்தும் ஏற்ற முடியவில்லை — "
        "விவரங்களுக்கு சேவையக பதிவுகளைச் சரிபார்க்கவும்.",
    "Model loaded": "மாதிரி ஏற்றப்பட்டது",
    "with {n} classes:": "{n} வகைகளுடன்:",
    "1 · Upload leaf image": "1 · இலைப் படத்தைப் பதிவேற்றவும்",
    "Leaf image (JPG / PNG)": "இலைப் படம் (JPG / PNG)",
    "Uploaded leaf": "பதிவேற்றப்பட்ட இலை",
    "Drop a clear, well-lit photo of a single leaf here.":
        "ஒரே ஒரு இலையின் தெளிவான, நல்ல வெளிச்சமுள்ள புகைப்படத்தை இங்கே விடவும்.",
    "Advanced options": "மேம்பட்ட விருப்பங்கள்",
    "Confidence threshold": "நம்பகத்தன்மை வரம்பு",
    "Predictions below this confidence are flagged as uncertain.":
        "இந்த நம்பகத்தன்மைக்குக் கீழ் உள்ள முன்னறிவிப்புகள் உறுதியற்றதாகக் குறிக்கப்படும்.",
    "Preprocessing": "முன்செயலாக்கம்",
    "Noise reduction": "இரைச்சல் குறைப்பு",
    "Apply OpenCV non-local-means denoising before inference. "
    "Useful for grainy or low-light photos.":
        "அனுமானத்திற்கு முன் OpenCV non-local-means இரைச்சல் நீக்கத்தைப் பயன்படுத்தவும். "
        "தானியமான அல்லது குறைந்த வெளிச்ச புகைப்படங்களுக்குப் பயனுள்ளது.",
    "Background handling": "பின்னணி கையாளுதல்",
    "Softly flatten non-leaf-colored background toward neutral "
    "gray so the model focuses on the leaf. Useful for busy "
    "backgrounds; skip for close-up leaf-only photos.":
        "மாதிரி இலையில் கவனம் செலுத்த, இலை நிறமற்ற பின்னணியை நடுநிலை சாம்பல் நிறமாக "
        "மென்மையாக்கவும். பரபரப்பான பின்னணிகளுக்குப் பயனுள்ளது; இலை மட்டும் கொண்ட "
        "நெருக்கமான புகைப்படங்களுக்குத் தவிர்க்கவும்.",
    "Analyze": "பகுப்பாய்வு செய்யவும்",
    "2 · Prediction result": "2 · முன்னறிவிப்பு முடிவு",
    "Preprocessing image…": "படம் முன்செயலாக்கப்படுகிறது…",
    "Running disease detection…": "நோய் கண்டறிதல் இயக்கப்படுகிறது…",
    "Computing explainability heatmap…": "விளக்கமளிக்கும் வெப்ப வரைபடம் கணக்கிடப்படுகிறது…",
    "Couldn't generate the explainability heatmap for this prediction.":
        "இந்த முன்னறிவிப்புக்கான விளக்கமளிக்கும் வெப்ப வரைபடத்தை உருவாக்க முடியவில்லை.",
    "Analyzing this image failed unexpectedly. Please try "
    "again. If the problem continues, contact the app maintainer.":
        "இந்த படத்தை பகுப்பாய்வு செய்வதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Awaiting analysis": "பகுப்பாய்வுக்காகக் காத்திருக்கிறது",
    "Upload an image and click **Analyze** to see the prediction, "
    "confidence, severity, and recommendation.":
        "படத்தைப் பதிவேற்றி, முன்னறிவிப்பு, நம்பகத்தன்மை, தீவிரம் மற்றும் "
        "பரிந்துரையைப் பார்க்க **பகுப்பாய்வு செய்யவும்** என்பதைக் கிளிக் செய்யவும்.",
    "Detected condition": "கண்டறியப்பட்ட நிலை",
    "Confidence": "நம்பகத்தன்மை",
    "model output": "மாதிரி வெளியீடு",
    "Severity": "தீவிரம்",
    "Threshold": "வரம்பு",
    "cutoff for reliable result": "நம்பகமான முடிவுக்கான வரம்பு",
    "Confidence is below the threshold. "
    "The result may be uncertain — consider retaking the photo with "
    "better lighting/focus.":
        "நம்பகத்தன்மை வரம்புக்குக் கீழ் உள்ளது. முடிவு உறுதியற்றதாக இருக்கலாம் — "
        "சிறந்த வெளிச்சம்/கவனத்துடன் புகைப்படத்தை மீண்டும் எடுக்க பரிசீலிக்கவும்.",
    "Confidence breakdown by class": "வகைவாரியான நம்பகத்தன்மை பிரிவு",
    "Confidence (%)": "நம்பகத்தன்மை (%)",
    "threshold": "வரம்பு",
    "Recommendation": "பரிந்துரை",
    "Estimated yield loss if untreated": "சிகிச்சை அளிக்கவில்லை எனில் மதிப்பிடப்பட்ட மகசூல் இழப்பு",
    "Based on published agricultural research for {disease} at "
    "{severity} severity. Adjust the figures below to your own field.":
        "{disease} நோய்க்கு {severity} தீவிரத்தில் வெளியிடப்பட்ட விவசாய ஆராய்ச்சியின் "
        "அடிப்படையில். கீழேயுள்ள எண்களை உங்கள் சொந்த வயலுக்கு ஏற்ப மாற்றவும்.",
    "Field size unit": "வயல் அளவு அலகு",
    "Hectares": "ஹெக்டேர்",
    "Acres": "ஏக்கர்",
    "Field size ({unit})": "வயல் அளவு ({unit})",
    "hectares": "ஹெக்டேர்",
    "acres": "ஏக்கர்",
    "Expected yield (t/ha if healthy)": "எதிர்பார்க்கப்படும் மகசூல் (ஆரோக்கியமாக இருந்தால் டன்/ஹெக்டேர்)",
    "Pre-filled with a rough global reference for this crop — "
    "replace with your farm's typical yield for a more accurate estimate.":
        "இந்த பயிருக்கான தோராயமான உலகளாவிய குறிப்புடன் முன்கூட்டியே நிரப்பப்பட்டுள்ளது — "
        "மிகச் சரியான மதிப்பீட்டிற்கு உங்கள் பண்ணையின் வழக்கமான மகசூலுடன் மாற்றவும்.",
    "Price per tonne (optional)": "ஒரு டன்னுக்கான விலை (விருப்பத்திற்குரியது)",
    "Leave at 0 to see only the yield-loss estimate, with no revenue figure.":
        "வருவாய் எண் இல்லாமல் மகசூல்-இழப்பு மதிப்பீட்டை மட்டும் காண 0-ல் விடவும்.",
    "estimated revenue at risk": "ஆபத்தில் உள்ள மதிப்பிடப்பட்ட வருவாய்",
    "yield loss": "மகசூல் இழப்பு",
    "on your": "உங்கள்",
    "field (expected": "வயலில் (எதிர்பார்க்கப்படும்",
    "t if healthy)": "டன் ஆரோக்கியமாக இருந்தால்)",
    "Planning estimate from published crop-disease research, not a guarantee — actual "
    "loss depends on variety, timing of infection, weather, and management. Not financial advice.":
        "வெளியிடப்பட்ட பயிர்-நோய் ஆராய்ச்சியிலிருந்து திட்டமிடல் மதிப்பீடு, உத்தரவாதம் அல்ல — "
        "உண்மையான இழப்பு வகை, தொற்றின் நேரம், வானிலை மற்றும் மேலாண்மையைப் பொறுத்தது. "
        "இது நிதி ஆலோசனை அல்ல.",
    "Why this prediction? (Grad-CAM)": "இந்த முன்னறிவிப்பு ஏன்? (Grad-CAM)",
    "Explainability heatmap unavailable for this prediction.":
        "இந்த முன்னறிவிப்புக்கான விளக்கமளிக்கும் வெப்ப வரைபடம் கிடைக்கவில்லை.",
    "Heatmap intensity": "வெப்ப வரைபட தீவிரம்",
    "How strongly the heatmap is blended over the leaf image below. "
    "This only re-blends the already-computed heatmap — it does not "
    "re-run the model.":
        "வெப்ப வரைபடம் கீழே உள்ள இலைப் படத்தின் மீது எவ்வளவு தீவிரமாக கலக்கப்படுகிறது. "
        "இது ஏற்கனவே கணக்கிடப்பட்ட வெப்ப வரைபடத்தை மட்டுமே மீண்டும் கலக்கிறது — "
        "மாதிரியை மீண்டும் இயக்காது.",
    "What the model saw (224×224 input)": "மாதிரி பார்த்தது (224×224 உள்ளீடு)",
    "Grad-CAM for": "Grad-CAM க்கானது",
    "Warmer regions (red/yellow) contributed most to the prediction above;"
    " cooler regions (blue) contributed least. Computed by backpropagating"
    " the predicted class score to the model's last convolutional layer"
    " (Grad-CAM, Selvaraju et al. 2017).":
        "வெப்பமான பகுதிகள் (சிவப்பு/மஞ்சள்) மேலே உள்ள முன்னறிவிப்புக்கு அதிகம் பங்களித்தன; "
        "குளிர்ந்த பகுதிகள் (நீலம்) குறைவாக பங்களித்தன. கணிக்கப்பட்ட வகை மதிப்பெண்ணை "
        "மாதிரியின் கடைசி convolutional அடுக்குக்கு பின்-பரப்புவதன் மூலம் கணக்கிடப்பட்டது "
        "(Grad-CAM, Selvaraju et al. 2017).",
    "Download PDF Report": "PDF அறிக்கையைப் பதிவிறக்கவும்",
    "Couldn't generate the PDF report right now. Please try again.":
        "இப்போது PDF அறிக்கையை உருவாக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",
    "Analysis saved to database (ID: {id}).": "பகுப்பாய்வு தரவுத்தளத்தில் சேமிக்கப்பட்டது (ID: {id}).",
    "Saved": "சேமிக்கப்பட்டது",
    "Save Analysis": "பகுப்பாய்வை சேமிக்கவும்",
    "Saving analysis…": "பகுப்பாய்வு சேமிக்கப்படுகிறது…",
    "Re-run": "மீண்டும் இயக்கவும்",
    "Crop": "பயிர்",
}

# ---------------------------------------------------------------------------
# Dashboard page (pages/dashboard.py)
# ---------------------------------------------------------------------------
DASHBOARD_TA: dict[str, str] = {
    "Dashboard": "கட்டுப்பாட்டு பலகை",
    "Statistics and trends across all crop, disease, and environmental analyses.":
        "அனைத்து பயிர், நோய் மற்றும் சுற்றுச்சூழல் பகுப்பாய்வுகளின் புள்ளிவிவரங்கள் மற்றும் போக்குகள்.",
    "crop(s) trending worse": "பயிர்(கள்) மோசமடைந்து வருகின்றன",
    "See the Outbreak Alerts page for details.": "விவரங்களுக்கு பரவல் எச்சரிக்கைகள் பக்கத்தைப் பார்க்கவும்.",
    "Crop Health": "பயிர் ஆரோக்கியம்",
    "Disease Detection": "நோய் கண்டறிதல்",
    "Field Scans": "வயல் ஆய்வுகள்",
    "Environmental": "சுற்றுச்சூழல்",
    "Loading dashboard data failed unexpectedly. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "கட்டுப்பாட்டு பலகை தரவை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "No crop health analyses saved yet. Run a "
    "calculation on the <b>Crop Health Analysis</b> page and click "
    "<b>Save Analysis</b> to populate this tab.":
        "இதுவரை பயிர் ஆரோக்கிய பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>பயிர் ஆரோக்கிய "
        "பகுப்பாய்வு</b> பக்கத்தில் ஒரு கணக்கீட்டை இயக்கி, இந்தத் தாவலை நிரப்ப "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "No disease detection analyses saved yet. "
    "Analyze a leaf image on the <b>Disease Detection</b> page and click "
    "<b>Save Analysis</b> to populate this tab.":
        "இதுவரை நோய் கண்டறிதல் பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>நோய் கண்டறிதல்</b> "
        "பக்கத்தில் ஒரு இலைப் படத்தை பகுப்பாய்வு செய்து, இந்தத் தாவலை நிரப்ப "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "No field scans saved yet. Run a batch "
    "scan on the <b>Field Scan</b> page and click <b>Save Field Scan</b> "
    "to populate this tab.":
        "இதுவரை வயல் ஆய்வுகள் சேமிக்கப்படவில்லை. <b>வயல் ஆய்வு</b> பக்கத்தில் ஒரு "
        "தொகுதி ஆய்வை இயக்கி, இந்தத் தாவலை நிரப்ப <b>வயல் ஆய்வை சேமிக்கவும்</b> "
        "என்பதைக் கிளிக் செய்யவும்.",
    "No environmental analyses saved yet. "
    "Assess a reading on the <b>Environmental Analysis</b> page and click "
    "<b>Save Analysis</b> to populate this tab.":
        "இதுவரை சுற்றுச்சூழல் பகுப்பாய்வுகள் சேமிக்கப்படவில்லை. <b>சுற்றுச்சூழல் "
        "பகுப்பாய்வு</b> பக்கத்தில் ஒரு அளவீட்டை மதிப்பிட்டு, இந்தத் தாவலை நிரப்ப "
        "<b>பகுப்பாய்வை சேமிக்கவும்</b> என்பதைக் கிளிக் செய்யவும்.",
    "Loading disease detection dashboard data failed unexpectedly. "
    "Please try again. If the problem continues, contact the app maintainer.":
        "நோய் கண்டறிதல் கட்டுப்பாட்டு பலகை தரவை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. "
        "மீண்டும் முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Loading field scan dashboard data failed unexpectedly. "
    "Please try again. If the problem continues, contact the app maintainer.":
        "வயல் ஆய்வு கட்டுப்பாட்டு பலகை தரவை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. "
        "மீண்டும் முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Loading environmental dashboard data failed unexpectedly. "
    "Please try again. If the problem continues, contact the app maintainer.":
        "சுற்றுச்சூழல் கட்டுப்பாட்டு பலகை தரவை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. "
        "மீண்டும் முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "Total Analyses": "மொத்த பகுப்பாய்வுகள்",
    "all-time": "எல்லா காலத்திலும்",
    "Healthy Plants": "ஆரோக்கியமான செடிகள்",
    "of total": "மொத்தத்தில்",
    "Diseased Plants": "நோய்வாய்ப்பட்ட செடிகள்",
    "Avg Health Score": "சராசரி ஆரோக்கிய மதிப்பெண்",
    "out of 100": "100-ல்",
    "High-Risk Cases": "அதிக ஆபத்து வழக்குகள்",
    "At Risk + Critical": "ஆபத்தில் + அபாயகரமானது",
    "Number of analyses": "பகுப்பாய்வுகளின் எண்ணிக்கை",
    "Disease distribution": "நோய் பரவல்",
    "Analyses per day": "நாளொன்றுக்கான பகுப்பாய்வுகள்",
    "Avg health score": "சராசரி ஆரோக்கிய மதிப்பெண்",
    "Date": "தேதி",
    "Analyses / day": "பகுப்பாய்வுகள் / நாள்",
    "Health score trend": "ஆரோக்கிய மதிப்பெண் போக்கு",
    "Crop": "பயிர்",
    "Status": "நிலை",
    "Crop-wise analysis": "பயிர்வாரியான பகுப்பாய்வு",
    "Risk distribution": "ஆபத்து பரவல்",
    "Healthy Leaves": "ஆரோக்கியமான இலைகள்",
    "Diseased Leaves": "நோய்வாய்ப்பட்ட இலைகள்",
    "Avg Confidence": "சராசரி நம்பகத்தன்மை",
    "model output": "மாதிரி வெளியீடு",
    "High Severity": "அதிக தீவிரம்",
    "cases flagged High": "அதிகம் எனக் குறிக்கப்பட்ட வழக்குகள்",
    "Avg confidence": "சராசரி நம்பகத்தன்மை",
    "Avg confidence (%)": "சராசரி நம்பகத்தன்மை (%)",
    "Confidence trend": "நம்பகத்தன்மை போக்கு",
    "Disease": "நோய்",
    "Severity distribution": "தீவிர பரவல்",
    "Total Scans": "மொத்த ஆய்வுகள்",
    "Leaves Scanned": "ஆய்வு செய்யப்பட்ட இலைகள்",
    "across all scans": "அனைத்து ஆய்வுகளிலும்",
    "Avg Healthy %": "சராசரி ஆரோக்கியம் %",
    "per scan": "ஒரு ஆய்வுக்கு",
    "Avg Field Score": "சராசரி வயல் மதிப்பெண்",
    "High-Risk Scans": "அதிக ஆபத்து ஆய்வுகள்",
    "Scans per day": "நாளொன்றுக்கான ஆய்வுகள்",
    "Avg field score": "சராசரி வயல் மதிப்பெண்",
    "Scans / day": "ஆய்வுகள் / நாள்",
    "Field health score trend": "வயல் ஆரோக்கிய மதிப்பெண் போக்கு",
    "Avg field health score by crop": "பயிர்வாரியான சராசரி வயல் ஆரோக்கிய மதிப்பெண்",
    "Number of scans": "ஆய்வுகளின் எண்ணிக்கை",
    "Dominant disease across scans": "ஆய்வுகளில் முதன்மை நோய்",
    "Aggregate severity breakdown (all scanned leaves)": "மொத்த தீவிர பிரிவு (அனைத்து ஆய்வு செய்யப்பட்ட இலைகள்)",
    "None detected": "எதுவும் கண்டறியப்படவில்லை",
    "Optimal Readings": "உகந்த அளவீடுகள்",
    "High-Risk Readings": "அதிக ஆபத்து அளவீடுகள்",
    "High + Critical": "அதிகம் + அபாயகரமானது",
    "Avg Model Confidence": "சராசரி மாதிரி நம்பகத்தன்மை",
    "trained risk model": "பயிற்சி பெற்ற ஆபத்து மாதிரி",
    "Risk level distribution": "ஆபத்து அளவு பரவல்",
    "Risk level": "ஆபத்து அளவு",
    "Logged factor ranges": "பதிவு செய்யப்பட்ட காரணி வரம்புகள்",
    "Value": "மதிப்பு",
    "Temp (°C)": "வெப்பநிலை (°C)",
    "Humidity (%)": "ஈரப்பதம் (%)",
    "Soil moisture (%)": "மண் ஈரப்பதம் (%)",
    "Rainfall (mm)": "மழைப்பொழிவு (mm)",
}

# ---------------------------------------------------------------------------
# Crop Doctor — rule-based Q&A over a saved analysis record
# (src/crop_doctor.py, via tr_template) + pages/crop_doctor.py UI strings
# ---------------------------------------------------------------------------
CROP_DOCTOR_TA: dict[str, str] = {
    # --- src/crop_doctor.py answer templates ---
    "I couldn't work that out from this record. Try one of the "
    "suggested questions below, or rephrase — e.g. \"why is this "
    "moderate severity?\"":
        "இந்த பதிவிலிருந்து அதைக் கண்டறிய முடியவில்லை. கீழே உள்ள பரிந்துரைக்கப்பட்ட "
        "கேள்விகளில் ஒன்றை முயற்சிக்கவும், அல்லது வேறுவிதமாகக் கேட்கவும் — உதாரணமாக "
        "\"இது ஏன் மிதமான தீவிரம்?\"",
    "This {kind} record doesn't have a disease/severity call to explain.":
        "இந்த {kind} பதிவில் விளக்க நோய்/தீவிர முடிவு இல்லை.",
    "'{disease}' was recorded at {severity} severity.":
        "'{disease}' {severity} தீவிரத்தில் பதிவு செய்யப்பட்டது.",
    "This {kind} record doesn't have a model confidence value stored.":
        "இந்த {kind} பதிவில் மாதிரி நம்பகத்தன்மை மதிப்பு சேமிக்கப்படவில்லை.",
    "The model was {pct:.0f}% confident in '{disease}' — that's a high-confidence call.":
        "மாதிரி '{disease}' என்பதில் {pct:.0f}% நம்பகத்தன்மை கொண்டிருந்தது — இது அதிக நம்பகத்தன்மையான முடிவு.",
    "The model was {pct:.0f}% confident in '{disease}' — moderate confidence, usually "
    "reliable, but a second clear photo wouldn't hurt if you want to be sure.":
        "மாதிரி '{disease}' என்பதில் {pct:.0f}% நம்பகத்தன்மை கொண்டிருந்தது — மிதமான "
        "நம்பகத்தன்மை, பொதுவாக நம்பகமானது, ஆனால் நிச்சயமாக அறிய விரும்பினால் இரண்டாவது "
        "தெளிவான புகைப்படம் எடுப்பது நல்லது.",
    "The model was only {pct:.0f}% confident in '{disease}' — that's on the lower side. "
    "Consider retaking the photo with better lighting and a closer, single-leaf shot for "
    "a more reliable result.":
        "மாதிரி '{disease}' என்பதில் {pct:.0f}% நம்பகத்தன்மை மட்டுமே கொண்டிருந்தது — இது "
        "குறைவான பக்கத்தில் உள்ளது. மிகவும் நம்பகமான முடிவுக்கு, சிறந்த வெளிச்சத்துடன் "
        "நெருக்கமான, ஒரே இலையின் புகைப்படத்தை மீண்டும் எடுக்க பரிசீலிக்கவும்.",
    "This record doesn't show a disease to leave untreated — it was healthy, or has "
    "no disease/severity call to reason about.":
        "இந்த பதிவில் சிகிச்சையளிக்காமல் விடுவதற்கான நோய் எதுவும் இல்லை — இது ஆரோக்கியமாக "
        "இருந்தது, அல்லது பகுத்தாய்வதற்கு நோய்/தீவிர முடிவு இல்லை.",
    "'{disease}' at {severity} severity isn't in the yield-loss reference table yet, "
    "so I can't give a data-backed estimate — but as a rule, delaying treatment on a "
    "spreading leaf disease rarely helps and usually costs more the longer it waits.":
        "'{disease}' {severity} தீவிரத்தில் இன்னும் மகசூல்-இழப்பு குறிப்பு அட்டவணையில் "
        "இல்லை, எனவே தரவு அடிப்படையிலான மதிப்பீட்டை தர முடியாது — ஆனால் ஒரு விதியாக, "
        "பரவும் இலை நோய்க்கு சிகிச்சையை தாமதப்படுத்துவது அரிதாகவே உதவும், மேலும் அது "
        "எவ்வளவு காத்திருக்கிறதோ அவ்வளவு அதிகமாக செலவாகும்.",
    "Published research on {disease} at {severity} severity suggests {low:.0f}-{high:.0f}% "
    "yield loss if left untreated under comparable conditions — on a 1-hectare field "
    "yielding {yield_per_ha:.1f} t/ha, that's roughly {yl_low:.1f}-{yl_high:.1f} tonnes. It "
    "typically doesn't improve on its own; treating promptly is the lower-risk choice.":
        "{disease} {severity} தீவிரத்தில் இருந்தால், சிகிச்சையளிக்காமல் விட்டால் "
        "ஒப்பிடத்தக்க நிலைமைகளின் கீழ் {low:.0f}-{high:.0f}% மகசூல் இழப்பு ஏற்படும் என "
        "வெளியிடப்பட்ட ஆராய்ச்சி கூறுகிறது — 1 ஹெக்டேர் வயலில் {yield_per_ha:.1f} டன்/ஹெக்டேர் "
        "மகசூல் எதிர்பார்க்கப்பட்டால், அது தோராயமாக {yl_low:.1f}-{yl_high:.1f} டன்கள். இது "
        "பொதுவாக தானாக மேம்படாது; உடனடியாக சிகிச்சையளிப்பதே குறைந்த ஆபத்துள்ள தேர்வு.",
    "No disease was detected on this record, so there's nothing here that would spread.":
        "இந்த பதிவில் எந்த நோயும் கண்டறியப்படவில்லை, எனவே இங்கே பரவக்கூடியது எதுவும் இல்லை.",
    "Yes — {disease} is known to spread quickly, especially in humid or wet conditions. "
    "Isolate or remove affected material where possible and treat promptly to limit it "
    "reaching nearby plants.":
        "ஆம் — {disease} விரைவாக பரவும் என அறியப்படுகிறது, குறிப்பாக ஈரப்பதமான அல்லது "
        "நனைந்த நிலைமைகளில். முடிந்தால் பாதிக்கப்பட்ட பகுதிகளை தனிமைப்படுத்தவும் அல்லது "
        "அகற்றவும், அருகிலுள்ள செடிகளை அடையாமல் தடுக்க உடனடியாக சிகிச்சையளிக்கவும்.",
    "{disease} can spread under favorable (humid/warm) conditions, but usually more "
    "slowly than blast/blight-type diseases. Regular monitoring and timely treatment "
    "should keep it contained.":
        "{disease} சாதகமான (ஈரப்பதம்/வெப்பமான) நிலைமைகளின் கீழ் பரவக்கூடும், ஆனால் "
        "பொதுவாக blast/blight வகை நோய்களை விட மெதுவாக. வழக்கமான கண்காணிப்பும் "
        "சரியான நேரத்தில் சிகிச்சையும் அதை கட்டுப்பாட்டில் வைத்திருக்கும்.",
    "I don't have a specific spread-risk rule for '{disease}' yet, but as a general "
    "precaution, monitor nearby plants and treat promptly regardless.":
        "'{disease}' க்கான குறிப்பிட்ட பரவல்-ஆபத்து விதி இன்னும் என்னிடம் இல்லை, ஆனால் "
        "பொதுவான முன்னெச்சரிக்கையாக, அருகிலுள்ள செடிகளை கண்காணித்து, எப்படியிருந்தாலும் "
        "உடனடியாக சிகிச்சையளிக்கவும்.",
    "No disease was detected here, but these readings were outside {crop}'s "
    "ideal range, which is worth watching:\n{bullets}":
        "இங்கே எந்த நோயும் கண்டறியப்படவில்லை, ஆனால் இந்த அளவீடுகள் {crop}-ன் "
        "உகந்த வரம்புக்கு வெளியே இருந்தன, இதைக் கவனிக்க வேண்டும்:\n{bullets}",
    "No disease was detected on this record, and the logged readings (if any) were "
    "within range — nothing here points to a cause because there's no problem to "
    "explain.":
        "இந்த பதிவில் எந்த நோயும் கண்டறியப்படவில்லை, மேலும் பதிவு செய்யப்பட்ட அளவீடுகள் "
        "(ஏதேனும் இருந்தால்) வரம்பிற்குள் இருந்தன — விளக்குவதற்கு சிக்கல் எதுவும் "
        "இல்லாததால் இங்கே எந்த காரணத்தையும் சுட்டிக்காட்டவில்லை.",
    "'{disease}' was detected at {severity} severity. These conditions were outside "
    "{crop}'s ideal range and likely contributed:\n{bullets}":
        "'{disease}' {severity} தீவிரத்தில் கண்டறியப்பட்டது. இந்த நிலைமைகள் "
        "{crop}-ன் உகந்த வரம்புக்கு வெளியே இருந்தன, மேலும் அவை பங்களித்திருக்கலாம்:\n{bullets}",
    "'{disease}' was detected at {severity} severity. The logged environmental "
    "readings were within {crop}'s typical range, so the immediate cause here is the "
    "leaf-image classification itself, not an obvious environmental trigger.":
        "'{disease}' {severity} தீவிரத்தில் கண்டறியப்பட்டது. பதிவு செய்யப்பட்ட "
        "சுற்றுச்சூழல் அளவீடுகள் {crop}-ன் வழக்கமான வரம்புக்குள் இருந்தன, எனவே இங்கு "
        "உடனடி காரணம் இலைப்-பட வகைப்பாடே தவிர, தெளிவான சுற்றுச்சூழல் தூண்டுதல் அல்ல.",
    "'{disease}' was detected at {severity} severity from the leaf image alone — this "
    "record doesn't have environmental readings, so I can't say whether conditions "
    "contributed. Check Environmental Analysis for this crop around the same date.":
        "'{disease}' {severity} தீவிரத்தில் இலைப் படத்தில் இருந்து மட்டுமே கண்டறியப்பட்டது — "
        "இந்த பதிவில் சுற்றுச்சூழல் அளவீடுகள் இல்லை, எனவே நிலைமைகள் பங்களித்தனவா என்று "
        "சொல்ல முடியாது. அதே தேதியில் இந்த பயிருக்கான சுற்றுச்சூழல் பகுப்பாய்வைப் பார்க்கவும்.",
    "This {kind} record doesn't have a health score stored.":
        "இந்த {kind} பதிவில் ஆரோக்கிய மதிப்பெண் சேமிக்கப்படவில்லை.",
    "The health score is {score}/100 ('{status}'), combining a disease-signal risk of "
    "'{disease_risk}' and an environmental-signal risk of '{env_risk}' — disease is "
    "weighed slightly more heavily (55%) than environment (45%) in this blend.":
        "ஆரோக்கிய மதிப்பெண் {score}/100 ('{status}'), நோய்-சமிக்ஞை ஆபத்து '{disease_risk}' "
        "மற்றும் சுற்றுச்சூழல்-சமிக்ஞை ஆபத்து '{env_risk}' ஆகியவற்றை இணைக்கிறது — இந்த "
        "கலவையில் சுற்றுச்சூழலை (45%) விட நோய் சற்று அதிகமாக (55%) கணக்கில் எடுத்துக்கொள்ளப்படுகிறது.",
    "The health score is {score}/100, which falls in the '{status}' band.":
        "ஆரோக்கிய மதிப்பெண் {score}/100, இது '{status}' பிரிவில் வருகிறது.",
    "This {kind} record doesn't have environmental readings stored.":
        "இந்த {kind} பதிவில் சுற்றுச்சூழல் அளவீடுகள் சேமிக்கப்படவில்லை.",
    "Logged readings for this analysis:\n{factors}\n\nOutside {crop}'s ideal range:\n{issues}":
        "இந்த பகுப்பாய்வுக்கான பதிவு செய்யப்பட்ட அளவீடுகள்:\n{factors}\n\n{crop}-ன் உகந்த "
        "வரம்புக்கு வெளியே:\n{issues}",
    "Logged readings for this analysis:\n{factors}\n\nAll within {crop}'s typical range.":
        "இந்த பகுப்பாய்வுக்கான பதிவு செய்யப்பட்ட அளவீடுகள்:\n{factors}\n\nஅனைத்தும் "
        "{crop}-ன் வழக்கமான வரம்புக்குள் உள்ளன.",
    "{summary}\n\nTop priority actions:\n{bullets}":
        "{summary}\n\nமுன்னுரிமை நடவடிக்கைகள்:\n{bullets}",
    "No recommendation was stored with this record.":
        "இந்த பதிவுடன் எந்த பரிந்துரையும் சேமிக்கப்படவில்லை.",
    "I can answer questions about this saved {kind} analysis — its severity, how "
    "confident the model was, what happens if it's left untreated, spread risk, its "
    "recommendation, and (where logged) the environmental readings. Try one of the "
    "suggested questions below, or ask in your own words.":
        "இந்த சேமிக்கப்பட்ட {kind} பகுப்பாய்வு பற்றிய கேள்விகளுக்கு என்னால் பதிலளிக்க "
        "முடியும் — அதன் தீவிரம், மாதிரி எவ்வளவு நம்பகமாக இருந்தது, சிகிச்சையளிக்காமல் "
        "விட்டால் என்ன நடக்கும், பரவல் ஆபத்து, அதன் பரிந்துரை, மற்றும் (பதிவு "
        "செய்யப்பட்டிருந்தால்) சுற்றுச்சூழல் அளவீடுகள். கீழே உள்ள பரிந்துரைக்கப்பட்ட "
        "கேள்விகளில் ஒன்றை முயற்சிக்கவும், அல்லது உங்கள் சொந்த வார்த்தைகளில் கேளுங்கள்.",

    # --- record kind labels (used as {kind} above) ---
    "Crop Health": "பயிர் ஆரோக்கியம்",
    "Disease Detection": "நோய் கண்டறிதல்",
    "Environmental": "சுற்றுச்சூழல்",
    "Field Scan": "வயல் ஆய்வு",
    "analysis": "பகுப்பாய்வு",
    "this crop": "இந்த பயிர்",
    "this result": "இந்த முடிவு",
    "Unknown": "தெரியவில்லை",
    "None detected": "எதுவும் கண்டறியப்படவில்லை",
    "healthy": "ஆரோக்கியமானது",

    # --- pages/crop_doctor.py UI ---
    "Crop Doctor": "பயிர் மருத்துவர்",
    "Ask the Crop Doctor": "பயிர் மருத்துவரிடம் கேளுங்கள்",
    "Ask why a result was flagged, what happens if untreated, and get "
    "answers grounded in that saved analysis.":
        "ஒரு முடிவு ஏன் குறிக்கப்பட்டது, சிகிச்சையளிக்காவிட்டால் என்ன நடக்கும் என்று "
        "கேளுங்கள், மேலும் அந்த சேமிக்கப்பட்ட பகுப்பாய்வை அடிப்படையாகக் கொண்ட "
        "பதில்களைப் பெறுங்கள்.",
    "Ask questions about one of your saved analyses — answers are grounded "
    "in that record's actual numbers, not a generic chatbot.":
        "உங்கள் சேமிக்கப்பட்ட பகுப்பாய்வுகளில் ஒன்றைப் பற்றி கேள்விகள் கேளுங்கள் — பதில்கள் "
        "பொதுவான chatbot அல்ல, அந்த பதிவின் உண்மையான எண்களை அடிப்படையாகக் கொண்டவை.",
    "1 · Pick a saved analysis": "1 · சேமிக்கப்பட்ட பகுப்பாய்வைத் தேர்ந்தெடுக்கவும்",
    "Analysis type": "பகுப்பாய்வு வகை",
    "Record": "பதிவு",
    "No saved analyses of this type yet. Save one from the relevant page first.":
        "இந்த வகையின் சேமிக்கப்பட்ட பகுப்பாய்வுகள் இதுவரை இல்லை. முதலில் தொடர்புடைய "
        "பக்கத்திலிருந்து ஒன்றை சேமிக்கவும்.",
    "Loading saved analyses failed unexpectedly. Please try again. "
    "If the problem continues, contact the app maintainer.":
        "சேமிக்கப்பட்ட பகுப்பாய்வுகளை ஏற்றுவதில் எதிர்பாராத பிழை ஏற்பட்டது. மீண்டும் "
        "முயற்சிக்கவும். சிக்கல் தொடர்ந்தால், செயலி நிர்வாகியை தொடர்பு கொள்ளவும்.",
    "2 · Ask a question": "2 · ஒரு கேள்வி கேளுங்கள்",
    "Try asking": "கேட்டுப் பாருங்கள்",
    "Ask about this analysis…": "இந்த பகுப்பாய்வைப் பற்றி கேளுங்கள்…",
    "New conversation": "புதிய உரையாடல்",
    "Grounded in": "இதன் அடிப்படையில்",
    "This app never calls an external AI chat service for this — every "
    "answer is generated locally from this app's own rules and stored data.":
        "இதற்காக இந்த செயலி ஒருபோதும் வெளிப்புற AI chat சேவையை அழைக்காது — ஒவ்வொரு "
        "பதிலும் இந்த செயலியின் சொந்த விதிகள் மற்றும் சேமிக்கப்பட்ட தரவிலிருந்து "
        "உள்ளூரில் உருவாக்கப்படுகிறது.",

    # --- suggested quick-question chips (src/crop_doctor.py SUGGESTED_QUESTIONS) ---
    "Why is this severity level?": "இது ஏன் இந்த தீவிர நிலை?",
    "What if I don't treat it?": "நான் சிகிச்சையளிக்கவில்லை என்றால் என்ன ஆகும்?",
    "What should I do now?": "நான் இப்போது என்ன செய்ய வேண்டும்?",
    "Why is the health score what it is?": "ஆரோக்கிய மதிப்பெண் ஏன் இப்படி உள்ளது?",
    "How confident are you?": "நீங்கள் எவ்வளவு நம்பகத்தன்மையுடன் இருக்கிறீர்கள்?",
    "Will it spread?": "இது பரவுமா?",
    "What's out of range here?": "இங்கே எது வரம்புக்கு வெளியே உள்ளது?",
}


# ---------------------------------------------------------------------------
# Lookup helpers — every one falls back to the English source on a miss,
# so an untranslated string never breaks the page, it just stays English.
# ---------------------------------------------------------------------------
def tr_crop(crop: str, lang: str) -> str:
    if lang != "ta":
        return crop
    return CROP_NAMES_TA.get(crop, crop)


def tr_disease(disease: str, lang: str) -> str:
    """Translated disease display name. Falls back to the English pretty
    name (underscores -> spaces) if this exact class isn't in the table.
    """
    if lang != "ta":
        return disease.replace("_", " ")
    return DISEASE_NAMES_TA.get(disease, disease.replace("_", " "))


def tr_severity(severity: str, lang: str) -> str:
    if lang != "ta":
        return severity
    return SEVERITY_LABELS_TA.get(severity, severity)


def tr_severity_action(action_text: str, lang: str) -> str:
    if lang != "ta":
        return action_text
    return SEVERITY_ACTION_TA.get(action_text, action_text)


def tr_recommendation(disease: str, lang: str, fallback: str = "") -> str:
    """Translated recommendation text for a disease class. Falls back to
    `fallback` (typically the English RECOMMENDATION_MAP text the caller
    already has) if this disease isn't in the Tamil table yet.
    """
    if lang != "ta":
        return fallback
    return RECOMMENDATION_TA.get(disease, fallback)


def tr_env_tip(factor_key: str, status: str, crop: str, lang: str, fallback: str = "") -> str:
    """Translated environmental advice tip for one factor+status, with the
    crop name filled in (and itself translated). Falls back to `fallback`
    (the caller's already-formatted English tip) if this exact
    factor/status isn't in the table.
    """
    if lang != "ta":
        return fallback
    template = ENV_TIPS_TA.get(factor_key, {}).get(status)
    if not template:
        return fallback
    return template.format(crop=tr_crop(crop, lang))


def tr_env_all_optimal(crop: str, lang: str, fallback: str = "") -> str:
    if lang != "ta":
        return fallback
    return ENV_ADVICE_ALL_OPTIMAL_TA.format(crop=tr_crop(crop, lang))


def tr_template(english_template: str, lang: str, **kwargs) -> str:
    """Translate a sentence TEMPLATE (with {placeholder}s) and fill it in,
    for the dynamic explanation/recommendation generators (health_engine,
    environment_model, outbreak_detection, recommendation_engine) whose
    output includes live numbers/names, not just fixed lookups.

    Falls back to formatting the English template itself if no Tamil
    version exists yet — always returns a valid, filled-in sentence.
    """
    if lang == "ta" and english_template in HEALTH_ENGINE_TA:
        template = HEALTH_ENGINE_TA[english_template]
    elif lang == "ta" and english_template in REC_MISC_TA:
        template = REC_MISC_TA[english_template]
    elif lang == "ta" and english_template in OUTBREAK_TA:
        template = OUTBREAK_TA[english_template]
    elif lang == "ta" and english_template in HISTORY_TA:
        template = HISTORY_TA[english_template]
    elif lang == "ta" and english_template in FIELD_SCAN_TA:
        template = FIELD_SCAN_TA[english_template]
    elif lang == "ta" and english_template in DISEASE_PAGE_TA:
        template = DISEASE_PAGE_TA[english_template]
    elif lang == "ta" and english_template in DASHBOARD_TA:
        template = DASHBOARD_TA[english_template]
    elif lang == "ta" and english_template in CROP_DOCTOR_TA:
        template = CROP_DOCTOR_TA[english_template]
    else:
        template = english_template
    return template.format(**kwargs)


def tr_rec_keyword(keyword: str, lang: str, fallback: str = "") -> str:
    if lang != "ta":
        return fallback
    return REC_KEYWORD_ADVICE_TA.get(keyword, fallback)


def tr_rec_keyword_label(keyword: str, lang: str) -> str:
    if lang != "ta":
        return keyword
    return REC_KEYWORD_LABEL_TA.get(keyword, keyword)


def tr_rec_severity_action(text: str, lang: str) -> str:
    if lang != "ta":
        return text
    return REC_SEVERITY_ACTION_TA.get(text, text)


def tr_rec_env_advice(factor_key: str, status: str, lang: str, fallback: str = "") -> str:
    if lang != "ta":
        return fallback
    return REC_ENV_ADVICE_TA.get(factor_key, {}).get(status, fallback)


def tr_rec_health_score(band_index: int, lang: str, fallback: str = "") -> str:
    if lang != "ta" or band_index >= len(REC_HEALTH_SCORE_TA):
        return fallback
    return REC_HEALTH_SCORE_TA[band_index]


def tr_label(text: str, lang: str) -> str:
    """Translated UI label — checks every label table in turn (UI_LABELS_TA,
    NAV_LABELS_TA, HEALTH_STATUS_TA, RISK_LEVEL_TA, MISC_TA). One function,
    so call sites never need to know which table a given string lives in.
    """
    if lang != "ta":
        return text
    for table in (UI_LABELS_TA, NAV_LABELS_TA, HEALTH_STATUS_TA, RISK_LEVEL_TA, MISC_TA, HOME_TA,
                  ENV_UI_TA, ENV_FACTOR_LABELS_TA, ENV_STATUS_TA, ENV_NOTE_TA, HEALTH_ENGINE_TA, REC_MISC_TA,
                  HEALTH_PAGE_TA, ABOUT_TA, ALERTS_TA, OUTBREAK_TA, HISTORY_TA, FIELD_SCAN_TA,
                  DISEASE_PAGE_TA, DASHBOARD_TA, CROP_DOCTOR_TA):
        if text in table:
            return table[text]
    return text


# Short alias — same function, shorter name for call sites translating many
# small strings inline (e.g. table headers, chart labels).
tr = tr_label


# ---------------------------------------------------------------------------
# Session-state-backed current-language helper (Streamlit-aware, but kept
# as a thin optional layer so the tr_*() functions above stay pure and
# testable without a Streamlit session).
# ---------------------------------------------------------------------------
def get_language() -> str:
    import streamlit as st
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def set_language(lang: str) -> None:
    import streamlit as st
    st.session_state["language"] = lang


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Coverage check: every disease name has a recommendation and vice versa ---")
    disease_keys = set(DISEASE_NAMES_TA.keys())
    rec_keys = set(RECOMMENDATION_TA.keys())
    print("In DISEASE_NAMES_TA but not RECOMMENDATION_TA:", disease_keys - rec_keys)
    print("In RECOMMENDATION_TA but not DISEASE_NAMES_TA:", rec_keys - disease_keys)
    assert disease_keys == rec_keys, "Disease name and recommendation tables should cover the same classes"

    print("\n--- Fallback behavior for an unknown class ---")
    assert tr_disease("Some_New_Disease", "ta") == "Some New Disease"
    assert tr_recommendation("Some_New_Disease", "ta", fallback="Fallback text.") == "Fallback text."
    print("OK — unknown classes fall back to English rather than breaking.")

    print("\n--- English passthrough ---")
    assert tr_disease("Late_Blight", "en") == "Late Blight"
    assert tr_crop("Tomato", "en") == "Tomato"
    print("OK — lang='en' always returns the English form untouched.")

    print("\n--- Sample Tamil output ---")
    for crop, disease in [("Tomato", "Late_Blight"), ("Rice", "Neck_Blast"), ("Corn", "Gray_Leaf_Spot")]:
        print(f"{tr_crop(crop, 'ta')} / {tr_disease(disease, 'ta')}: {tr_recommendation(disease, 'ta')}")

    print("\nOK — all i18n checks passed.")