# Detekcia značiek modelom MTCNN

Tento repozitár obsahuje výsledok mojej bakalárskej práce, detektor značiek založený na kaskádovom modeli 


## Závislosti

```
matplotlib==2.1.1
torch==1.0.1.post2
numpy==1.13.3
opencv_python==3.4.4.19
```

## Spustenie dema

V zlože code sa nachádza súbor demo.py, ktorý spustí detekciu nad 10 obrázkami zo zložky images. Pre každý z obrázkov zobrazí ohraničujúce boxy s číslom triedy. A na štandardný výstup pošle údaje z detekcie.

Spustenie:
```
python3 demo.py
```

## Výstup metódy detector

```
[
    [ lavá strana, vrch, pravá strana, spodok, skóre, trieda ],
     ...
] 
```

Príklad výstupu s dvomi detekovanými značkami triedy 11 a 18: 
```
[
 [ 624.89387858  321.44675446  687.22755337  382.69294405  0.99941278  11.]
 [ 460.03783107  385.29183022  491.26540947  417.17840438  0.90608937  18.]
]
```

POZOR !!!
V súčasnej implementácii ak nedetekuje žiadnu značku v prvej alebo druhej fáze, vývolá ```ValueError``` výnimku,  a to na riadkoch  113, 162 modulu mtcnn.py.

## Detekované triedy značiek
<table>
<tr></tr>
<tr><td>

|Číslo triedy |  Značka |
--- | --- | 
0 | maximálna povolená rýchlosť 20
1 | maximálna povolená rýchlosť 30
2 | maximálna povolená rýchlosť 50
3 | maximálna povolená rýchlosť 60
4 | maximálna povolená rýchlosť 70
5 | maximálna povolená rýchlosť 80
6 | koniec maximálnej povolenej rýchlosti 80
7 | maximálna povolená rýchlosť 100
8 | maximálna povolená rýchlosť 120
9 | zákaz predchádzania
10 | zákaz predchádzania pre nákladné automobily
11 | križovatka s vedlajšou cestou
12 | hlavná cesta
13 | daj prednosť v jazde
14 | stop
15 | zákaz vjazdu v oboch smeroch
16 | zákaz vjazdu nákladných aút
17 | zákaz vjazdu v tomto smere
18 | iné nebezpečenstvo
19 | zákruta vľavo
20 | zákruta vpravo

</td><td>
  
|Číslo triedy |  Značka |
--- | --- |
21 | dvojitá zákruta, prvá vľavo
22 | nerovnosť vozovky
23 | nebezpečenstvo šmyku
24 | zúžená vozovka sprava
25 | práca na ceste
26 | svetelné signály
27 | pozor, chodci
28 | pozor, deti
29 | pozor, cyklisti
30 | sneh alebo poľadovica
31 | pozor, zver
32 | koniec viacerých zákazov
33 | prikázaný smer jazdy vpravo
34 | prikázaný smer jazdy vľavo
35 | prikázaný smer jazdy priamo
36 | prikázaný smer jazdy priamo a vpravo
37 | prikázaný smer jazdy priamo a vľavo
38 | prikázaný smer obchádzania vpravo
39 | prikázaný smer obchádzania vľavo
40 | kruhový objazd
41 | koniec zákazu predchádzania
42 | koniec zákazu predchádzania pre nákladné automobily

</td></tr> </table>
