# LSTM
python -u main.py --model lstm --epochs 100 --pred_len 96 --optimizer adam --criterion mse --lradj type1 --iteration 5

python -u main.py --model lstm --epochs 100 --pred_len 96 --optimizer sgd --criterion mse --lradj type1

python -u main.py --model lstm --epochs 100 --pred_len 96 --optimizer rmsprop --criterion mse --lradj type2

python -u main.py --model lstm --epochs 100 --pred_len 336 --optimizer adam --criterion mse --lradj type1

python -u main.py --model lstm --epochs 100 --pred_len 336 --optimizer sgd --criterion mse --lradj type1

python -u main.py --model lstm --epochs 100 --pred_len 336 --optimizer rmsprop --criterion mse --lradj type2

# Transformer
python -u main.py --model transformer --epochs 100 --pred_len 96 --optimizer adam --criterion mse --lradj type1 --iteration 5

python -u main.py --model transformer --epochs 100 --pred_len 96 --optimizer sgd --criterion mse --lradj type1

python -u main.py --model transformer --epochs 100 --pred_len 96 --optimizer rmsprop --criterion mse --lradj type2

python -u main.py --model transformer --epochs 100 --pred_len 336 --optimizer adam --criterion mse --lradj type1

python -u main.py --model transformer --epochs 100 --pred_len 336 --optimizer sgd --criterion mse --lradj type1

python -u main.py --model transformer --epochs 100 --pred_len 336 --optimizer rmsprop --criterion mse --lradj type2

# SciNet ????
