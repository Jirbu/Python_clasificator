PROMPT PRO AI AGENTA – COLOR CONSISTENCY (PER-LIMB SAMPLING)

DŮLEŽITÉ:

V projektu již existuje:

pose detection (MediaPipe)
valid_pose
multi-person tracking (crop regions)
motion validation

Tuto logiku NEMĚŇ.

Cílem je přidat appearance validation založenou na barvě jednotlivých částí těla.

CÍL

Pro každou osobu vytvořit:

appearance model (barvy těla)

a použít ho pro:

validaci identity osoby
filtraci falešných detekcí
HLAVNÍ PRINCIP

Pro každou část těla:

vzorkuj barvu podél končetiny (ne bounding box)
ukládej historii (buffer)
porovnávej s aktuálním frame
VRSTVA 1 – DEFINICE ČÁSTÍ TĚLA

Použij tyto segmenty:

torso (shoulder_center → hip_center)
left upper arm (shoulder → elbow)
right upper arm
left thigh (hip → knee)
right thigh
VRSTVA 2 – VÝBĚR BODŮ NA SEGMENTU

Pro každý segment:

vezmi 2 krajní body (A, B)
spočítej 5 bodů na úsečce:

t = [0, 0.25, 0.5, 0.75, 1]

point_i = A + t * (B - A)
ale pouziji se jen ty vnitřní 3
VRSTVA 3 – VÝPOČET BARVY (LOKÁLNÍ ROI)

Pro každý bod:

vezmi oblast 5×5 pixelů (25 pixelů)
střed = point_i

Pro celý segment:

máš 5 bodů, ale pouziji se jen ty 3 vnitrni→ celkem 75 pixelů

Spočítej:

mean RGB color ze všech pixelů

VRSTVA 4 – KONVERZE DO HSV

RGB → HSV

Použij pouze:

H (hue)
S (saturation)

Ignoruj V (value)

VRSTVA 5 – BUFFER (HISTORIE)

Pro každou osobu a každý segment:

udržuj buffer velikosti 10

Ukládej pouze pokud:

valid_pose == True
AND pose_confidence > 0.8

Použij FIFO (klouzavé okno)

VRSTVA 6 – REFERENČNÍ BARVA

Pro každý segment:

spočítej průměr z bufferu:

H_avg, S_avg

VRSTVA 7 – AKTUÁLNÍ FRAME

Pro aktuální pose:

spočítej H_current, S_current pro každý segment
VRSTVA 8 – VÝPOČET ODCHYLKY

Pro každý segment:

delta_H = circular_distance(H_current, H_avg)

(POZOR: hue je kruhové → použij min(|a-b|, 360-|a-b|))

delta_S = abs(S_current - S_avg)

VRSTVA 9 – SCORE SEGMENTU

error_segment = wH * delta_H + wS * delta_S

Doporučení:

wH = 0.7
wS = 0.3

VRSTVA 10 – VÁHOVÁNÍ SEGMENTŮ

Použij váhy:

torso = 0.4
arms = 0.15
legs = 0.15

final_error = weighted average(error_segment)

VRSTVA 11 – ROZHODNUTÍ

if final_error < threshold:

→ appearance_valid = True

else:

→ appearance_valid = False

VRSTVA 12 – INTEGRACE

Použij jako filtr:

if valid_pose == True
AND appearance_valid == True:

accept detection

else:

reject detection
VRSTVA 13 – OPTIMALIZACE
používej pouze body, které už máš z pose
nepočítej žádné bounding boxy navíc
nepoužívej full image scan
pracuj pouze s malými 5×5 oblastmi
VRSTVA 14 – DEBUG

Zobraz:

sampled body na těle
barvy segmentů
final_error
CÍL IMPLEMENTACE
stabilní identita osoby v čase
eliminace falešných skeletonů
minimální výpočetní náročnost