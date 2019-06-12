# Info

#### Narzędzie do ekstrakcji cech

Powinno znajdować się w folderze ```folder_projektu/extract_features```.

#### Obrazki

Wczytywane są z folderu ```folder_projektu/img```. Tam też znajdują się pliki .haraff.sift.

#### Obliczanie cech / macierzy odległości

Dla dwóch obrazków nie jest konieczne każdorazowe obliczanie macierzy odległości między punktami kluczowymi oraz ekstrakcja cech przez program.

Dlatego w ```__main__``` linijka ```imgs.loadFiles()``` konieczna jest do wywołania dla pary obrazków tylko za pierwszym razem. Potem można ją zakomentować. 

Podobnie jest z ```imgs.computeDistanceMatrix()``` -- obliczy ona macierz odległości (zajmuje ok. 1 minuty) a następnie zapisze ją do pliku o nazwie **obrazek1_obrazek2_distances** w katalogu ```folder_projektu/dist```.

Przy następnym uruchomieniu ```imgs.computeDistanceMatrix()``` można zastąpić ```imgs.loadDistanceMatrix()```, żeby nie obliczać ponownie macierzy odległości dla tych samych obrazków.
