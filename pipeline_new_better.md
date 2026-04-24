POKUD existuje osoba z předchozího framu (TRACKING) NEBO mám uložený frozen crop (LOST):
→ vezmi crop region (bounding box + 40 % margin)
→ spusť CELÝ validační pipeline na tomto cropu

VALIDAČNÍ PIPELINE (platí pro crop i full frame):

POSE DETECTION
→ spusť pose algoritmus
→ získáš landmarks + confidence

IF žádná poza:
→ return FAIL

IF confidence < low_threshold:
→ return FAIL

POSE VALIDACE (základní filtr)
→ zkontroluj viditelnost (ramena, kyčle)
→ zkontroluj základní geometrii

IF extrémně nevalidní:
→ return FAIL

KINEMATIKA (EARLY FILTER)

IF osoba existovala v minulém frame:

→ vezmi predikovanou pozici z trackeru  

→ spočítej vzdálenost mezi predikcí a aktuální pozicí  

IF vzdálenost > MAX_ALLOWED:  
    → return FAIL  // mimo očekávanou oblast (velmi nepravděpodobné)  

ELSE:  
    → kinematic_score = podle vzdálenosti (menší = lepší)  

ELSE:
→ kinematic_score = neutral

MOTION VALIDACE (LOKÁLNÍ)

→ vezmi ROI pro části těla
→ zarovnej historické framy (lokální posun)
→ spočítej podobnost

→ motion_score

IF similarity vysoká (např > 0.7):
→ motion_penalty (malý, ne FAIL)

APPEARANCE VALIDACE (BARVA)
(jeste neni iplementovano, prompt je v barva.md)

→ spočítej HSV pro segmenty těla
→ porovnej s bufferem

→ appearance_error

→ převeď na appearance_score

KOMBINACE (HLAVNÍ ROZHODNUTÍ)

→ spočítej final_confidence z:

pose confidence
kinematic_score
motion_penalty
appearance_score

např:

final_confidence =
w1 * pose_conf

w2 * kinematic_score
w3 * (1 - motion_penalty)
w4 * appearance_score

IF final_confidence < FINAL_THRESHOLD:
→ return FAIL

ÚSPĚCH

→ return SUCCESS + pose

HLAVNÍ LOGIKA FRAME:

IF existuje TRACKING nebo LOST osoba:

result = run_pipeline(crop_region)  

IF SUCCESS:  
    → aktualizuj tracker  
    → aktualizuj crop  
    → person_present = TRUE  

ELSE:  
    → spusť pipeline na celém frame  

    result_full = run_pipeline(full_frame)  

    IF SUCCESS:  
        → aktualizuj tracker  
        → vytvoř nový crop  
        → person_present = TRUE  

    ELSE:  
        → tracker update bez detekce  

        IF tracker drží osobu (ghost):  
            → person_present = TRUE  
        ELSE:  
            → person_present = FALSE  
            → stav = LOST  
            → uložit frozen crop  

POKUD neexistuje žádná osoba:

→ spusť pipeline na celém frame

IF SUCCESS:
→ sleduj kandidáta přes čas

IF potvrzen ve 3 framch a zároveň je tam v tech 3 framech splneno že se objekt pohybuje (je tam urcite kinematics score) tak potom:
    → vytvoř novou osobu  
    → inicializuj tracker  
    → vytvoř crop  

KLASIFIKACE:

IF person_present == TRUE AND valid_pose == TRUE:
→ extrahuj features
→ temporal window

IF window plné:  
    → klasifikace  
    → jump detector  

ELSE:
→ skip

DŮLEŽITÉ PRINCIPY:

kinematika je early filtr (rychlé vyřazení nesmyslů)
motion a appearance jsou měkké penalizace
žádný single filtr (kromě extrémů) nerozhoduje
finální rozhodnutí dělá kombinace
vždy crop → pak full frame
tracker má poslední slovo



Visualizace:
pro kazdeho cloveka zobrazovat:
zlute oramovani -bounding box ktery je pouzit jako crop.
uprostred torsa cloveka bude bod odkud bude vypsano tento seznam:

Pose confidence
1-motion_similarity number
kinematics score
aperance score
final confidence
person present: bool



