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
    "Detected condition (uncertain)": "கண்டறியப்பட்ட நிலை (உறுதியற்றது)",
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
    "Uncertain": "உறுதியற்றது",
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
                  HEALTH_PAGE_TA):
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