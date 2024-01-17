# MLCourseProject
2023 Fall ML course final project.

## How to run
```shell
# Best LSTM
python -u main.py --model lstm --pred_len 96 --lr 0.0001 --epochs 500 --batch_size 128 --optimizer adam --iteration 5
python -u main.py --model lstm --pred_len 336 --lr 0.0001 --epochs 500 --batch_size 128 --optimizer adam --iteration 5

# Best Transformer
python -u main.py --model transformer --pred_len 96 --lr 0.0001 --epochs 500 --batch_size 32 --optimizer adam --iteration 5
python -u main.py --model transformer --pred_len 336 --lr 0.0001 --epochs 500 --batch_size 32 --optimizer adam --iteration 5

# Best STNet
python -u main.py --model transformer --pred_len 96 --lr 0.001 --epochs 500 --batch_size 64 --optimizer rmsprop --iteration 5
python -u main.py --model transformer --pred_len 336 --lr 0.001 --epochs 500 --batch_size 64 --optimizer rmsprop --iteration 5
```