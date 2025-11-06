# geoguesser-ann
Using artificial neural networks for guessing goelocation from street view images

**PLAN**

1. Pobrać gęstość zaludnienia w Polsce albo z EUROSTATu albo jakiejś polskiej instytucji (im gęstsza siatka, tym lepsza), zapisać, wyświetlic w notatniku.
2. Funkcja do próbkowania dwuwymiarowego rozkładu prawdopodobieństwa (czyli mapy zaludnienia), sprawdzenie czy rozkład sie zgadza na 10000 punktów
3. Spróbować google streetview API
4. Pobrac 10 tys obrazków z API ( i pilnowac by nie przekroczyć, żeby nie zapłacić)
5. Skopiowac najpopularniejszego resneta
6. (Augmentacja danych: jeden obrazek z różnymi obrotami kamery?)
7. Wytrenować reneta dla bezpośrednie regresji (lat,lon) (oczywiście znormalizowane do -1,1 w granicach Polski, softmax na końcu Resneta), zobaczyc co wyjdzie
8. Klasyfikacja: albo komórki adaptatywne jak w DeepGeo (2018), albo podział na województwa (z OSM), zobaczyc co wyjdzie
