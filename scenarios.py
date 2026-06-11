"""
Zero Gravity Agent — Demo & Trigger Scenarios
==============================================
Deterministic scenarios used for the Demo and the Trigger Test Lab.
"""

# ── DEMO: Patient Voice Scenarios ────────────────────────────────────────────
DEMO_SCENARIOS = {
    "er_chest": {
        "id": "er_chest", "label": "ER — Chest Pain", "specialty": "Emergency", "emoji": "🚨",
        "patient": "Maria, 54 — Venezuela", "voice": "Aoede", "color": "#ef4444",
        "dna": "Venezuelan woman, 54, severe distress, trembling voice, rapid breathing. Speaks fast out of fear.",
        "script": "Doctor, me siento muy mal, tengo calentura desde esta mañana y me late muy fuerte el corazón. Siento como una punzada aquí en el pecho que se va al brazo izquierdo. Y se me fue la vista un momento hace como media hora. Tengo miedo.",
        "english_script": "Doctor, I feel very bad, I have had a fever since this morning and my heart is beating very fast. I feel like a stabbing pain here in my chest that goes to my left arm. And my vision went dark for a moment about half an hour ago. I am scared.",
    },
    "pediatrics_fever": {
        "id": "pediatrics_fever", "label": "Pediatrics — Fever & Diarrhea", "specialty": "Pediatrics", "emoji": "👶",
        "patient": "Carlos, 32 — Mexico", "voice": "Charon", "color": "#f59e0b",
        "dna": "Mexican man, 32, worried father, calm but tense voice.",
        "script": "Doctora, mi niño tiene chorro desde ayer por la noche, como cinco veces ya. Y tiene calentura, como de 38 grados y medio. No quiere comer nada y está muy flojo. Tiene dos años.",
        "english_script": "Doctor, my child has had diarrhea since last night, like five times already. And he has a fever, of about 38.5 degrees. He doesn't want to eat anything and is very weak. He is two years old.",
    },
    "ob_movement": {
        "id": "ob_movement", "label": "Obstetrics — Fetal Movement", "specialty": "OB/GYN", "emoji": "🤰",
        "patient": "Rosa, 28 — Puerto Rico", "voice": "Aoede", "color": "#8b5cf6",
        "dna": "Puerto Rican woman, 28, pregnant, highly anxious, voice between frightened and hopeful.",
        "script": "Enfermera, estoy en la semana 36 y siento que el bebé no se mueve como antes. Antes me pateaba bastante y hoy casi no lo siento. También me duele la barriga, como presión aquí abajo.",
        "english_script": "Nurse, I am in week 36 and I feel that the baby is not moving like before. Before, he used to kick me a lot and today I can barely feel him. Also, my abdomen hurts, like a pressure down here.",
    },
    "mental_health_si": {
        "id": "mental_health_si", "label": "Mental Health — Suicidal Ideation", "specialty": "Mental Health", "emoji": "🧠",
        "patient": "Luis, 41 — Colombia", "voice": "Charon", "color": "#6b7280",
        "dna": "Colombian man, 41, flat and monotonous voice, speaks slowly. Severe depression.",
        "script": "A veces pienso que sería mejor si no estuviera aquí. Mi familia estaría mejor sin mí. Ya tengo un plan, tengo pastillas en casa guardadas. No sé para qué sigo viniendo al médico.",
        "english_script": "Sometimes I think it would be better if I weren't here. My family would be better off without me. I already have a plan, I have pills stored at home. I don't know why I keep coming to the doctor.",
    },
    "folk_susto": {
        "id": "folk_susto", "label": "Folk Illness — Fright (Susto)", "specialty": "Cultural Brokering", "emoji": "🌿",
        "patient": "Pedro, 38 — Guatemala", "voice": "Charon", "color": "#10b981",
        "dna": "Guatemalan man, 38, describes symptoms culturally, deliberate pacing.",
        "script": "Doctor, creo que me cayó un susto muy fuerte cuando vi ese accidente el mes pasado. Desde entonces no puedo dormir, me tiemblan las manos, y siento frío de huesos aunque haga calor.",
        "english_script": "Doctor, I think I was struck by a very strong fright (susto) when I saw that accident last month. Since then, I can't sleep, my hands shake, and I feel chills in my bones (frío de huesos) even if it's hot.",
    },
    "consent_surgery": {
        "id": "consent_surgery", "label": "Informed Consent — Surgery", "specialty": "Consent", "emoji": "📋",
        "patient": "Elena, 67 — Ecuador", "voice": "Aoede", "color": "#3b82f6",
        "dna": "Ecuadorian elderly woman, 67, confused, basic education, calm but insecure voice.",
        "script": "Me dijeron que tengo que firmar unos papeles pero yo no entendí bien qué me van a hacer. Sí, dije que sí a todo pero la verdad es que no entendí. ¿Me pueden explicar de nuevo? Tengo miedo de que me operen y no despertar.",
        "english_script": "They told me I have to sign some papers but I didn't understand well what they are going to do to me. Yes, I said yes to everything, but the truth is I didn't understand. Can you explain it to me again? I'm afraid they will operate on me and I won't wake up.",
    },
    "family_interference": {
        "id": "family_interference", "label": "Family Interference", "specialty": "Role Boundary", "emoji": "⚠️",
        "patient": "Grandmother, 72 — Dominican Republic", "voice": "Aoede", "color": "#f97316",
        "dna": "Dominican elderly woman, soft voice, speaks quietly as if afraid.",
        "script": "Mija, mi hijo me dijo que no le diga al médico que yo también tomo las pastillas del marido. Me duele mucho el estómago y no sé si es por eso. Por favor no le diga a mi hijo que le conté esto.",
        "english_script": "My child, my son told me not to tell the doctor that I also take my husband's pills. My stomach hurts a lot and I don't know if it's because of that. Please don't tell my son that I told you this.",
    },
    "er_seizure": {
        "id": "er_seizure", "label": "ER — Post-Ictal Seizure", "specialty": "Emergency", "emoji": "⚡",
        "patient": "Jorge, 29 — El Salvador", "voice": "Charon", "color": "#ec4899",
        "dna": "Salvadoran man, 29, confused and dazed post-seizure, speaks slowly and is disoriented.",
        "script": "Yo no sé qué pasó. Me dijeron que me dio un ataque pero yo no recuerdo nada. Siento la lengua mordida y me duele mucho la cabeza. Tengo epilepsia desde niño pero hace tres años no me daba. Creo que no tomé el medicamento ayer.",
        "english_script": "I don't know what happened. They told me I had a seizure but I don't remember anything. I feel my tongue is bitten and my head hurts a lot. I've had epilepsy since I was a child but I hadn't had a seizure in three years. I think I didn't take the medication yesterday.",
    },
}

# ── SEMAPHORE TRIGGER TEST LAB ───────────────────────────────────────────────
# Each test has an `expected_state`. Run it → watch the semaphore → UI marks ✓/✗.
# Use to measure & tune toward 99% state-switching accuracy.
TRIGGER_TESTS = {
    "tt_conduit": {
        "id": "tt_conduit", "label": "Baseline — Normal Pain", "specialty": "🚦 Trigger Test",
        "emoji": "🟢", "patient": "Test → CONDUIT", "voice": "Aoede", "color": "#10b981",
        "expected_state": "CONDUIT",
        "dna": "Latino patient, calm and clear voice.",
        "script": "Doctor, me duele la cabeza desde ayer en la mañana y tengo un poco de fiebre.",
        "english_script": "Doctor, my head hurts since yesterday morning and I have a bit of a fever.",
    },
    "tt_nonsense": {
        "id": "tt_nonsense", "label": "Nonsense — Unicorn", "specialty": "🚦 Trigger Test",
        "emoji": "🟡", "patient": "Test → CLARIFIER", "voice": "Charon", "color": "#f59e0b",
        "expected_state": "CLARIFIER",
        "dna": "Confused patient, speaks confidently but speaks incoherently (possible aphasia).",
        "script": "Mi unicornio no encendía porque el autobús iba en reversa, y entonces las ventanas se comieron el reloj azul.",
        "english_script": "My unicorn wouldn't turn on because the bus was in reverse, and then the windows ate the blue clock.",
    },
    "tt_ambiguous": {
        "id": "tt_ambiguous", "label": "Ambiguous — Sudden Chill", "specialty": "🚦 Trigger Test",
        "emoji": "🟡", "patient": "Test → CLARIFIER", "voice": "Aoede", "color": "#f59e0b",
        "expected_state": "CLARIFIER",
        "dna": "Elderly woman, worried, describes symptoms ambiguously.",
        "script": "Doctor, anoche me dio un aire y desde entonces no me siento bien de este lado.",
        "english_script": "Doctor, last night I caught a cold draft (un aire) and since then I don't feel well on this side.",
    },
    "tt_susto": {
        "id": "tt_susto", "label": "Folk — Fright (Susto)", "specialty": "🚦 Trigger Test",
        "emoji": "🟣", "patient": "Test → CULTURAL", "voice": "Charon", "color": "#8b5cf6",
        "expected_state": "CULTURAL_BROKER",
        "dna": "Central American man, speaks slowly, describes in cultural terms.",
        "script": "Doctor, yo creo que me cayó un susto muy fuerte cuando vi el accidente, desde ahí no duermo.",
        "english_script": "Doctor, I think I was struck by a very strong fright (susto) when I saw the accident, since then I don't sleep.",
    },
    "tt_empacho": {
        "id": "tt_empacho", "label": "Folk — Indigestion (Empacho)", "specialty": "🚦 Trigger Test",
        "emoji": "🟣", "patient": "Test → CULTURAL", "voice": "Aoede", "color": "#8b5cf6",
        "expected_state": "CULTURAL_BROKER",
        "dna": "Mexican mother, worried about her child.",
        "script": "Mi niño tiene empacho desde que comió, está bien llorón, creo que también tiene mal de ojo.",
        "english_script": "My child has indigestion (empacho) since he ate, he is very tearful, I think he also has evil eye (mal de ojo).",
    },
    "tt_researcher": {
        "id": "tt_researcher", "label": "Rare Illness — Sydenham's Chorea", "specialty": "🚦 Trigger Test",
        "emoji": "🔬", "patient": "Test → RESEARCHER", "voice": "Charon", "color": "#38bdf8",
        "expected_state": "RESEARCHER",
        "dna": "Patient reporting a rare diagnosis with an uncommon name.",
        "script": "El doctor del otro hospital me dijo que tengo el mal de San Vito, pero no entendí bien qué es eso.",
        "english_script": "The doctor from the other hospital told me that I have Saint Vitus' Dance (mal de San Vito), but I didn't quite understand what that is.",
    },
    "tt_advocate_si": {
        "id": "tt_advocate_si", "label": "Safety — Suicidal Ideation", "specialty": "🚦 Trigger Test",
        "emoji": "🔴", "patient": "Test → ADVOCATE", "voice": "Charon", "color": "#ef4444",
        "expected_state": "ADVOCATE",
        "dna": "Man with severe depression, flat and monotonous voice.",
        "script": "A veces pienso que sería mejor no estar aquí, ya tengo unas pastillas guardadas en la casa.",
        "english_script": "Sometimes I think it would be better not to be here, I already have some pills stored at home.",
    },
    "tt_advocate_dose": {
        "id": "tt_advocate_dose", "label": "Safety — Dosage Confusion", "specialty": "🚦 Trigger Test",
        "emoji": "🔴", "patient": "Test → ADVOCATE", "voice": "Aoede", "color": "#ef4444",
        "expected_state": "ADVOCATE",
        "dna": "Patient reporting dangerous dosage confusion.",
        "script": "El doctor dijo veinticinco miligramos pero yo entendí que eran cincuenta, ya me tomé dos.",
        "english_script": "The doctor said twenty-five milligrams but I understood it was fifty, I already took two.",
    },
}

# Combine all scenarios for easy access
DEMO_SCENARIOS.update(TRIGGER_TESTS)
