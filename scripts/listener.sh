!/bin/bash

curl --user "shakhzod:shakhzod2000" -X POST "http://localhost:8042/tools/execute-script" --data-binary @sendToModality.lua -v
storescp 106 -v -aet BREASTCANCERAI +xs -od  ..\\dicomRaw --sort-on-study-uid st 

