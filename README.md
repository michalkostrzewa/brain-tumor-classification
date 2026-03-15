python3 -m venv ai_env

source ai_env/bin/activate

pip install -r requirements.txt

przykładowe wywołanie funkcji
python3 trening_resnet_224.py

w pliku macierz_resnet.py dostosuj do danych na jakich uczyl sie model  
 transforms.Resize((512, 512)), # transforms.Resize((224, 224)),
python3 macierz_resnet.py
