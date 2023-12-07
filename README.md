# MLCourseProject
2023 Fall ML course final project.

# 运行记录
epoch=100, lr=0.001, batch_size=64
1. 未进行数据归一化
```text
/home/mysong/.conda/envs/torch2/bin/python3 /tmp/pycharm_project_400/main.py 
Namespace(O=96, batch_size=64, dev=device(type='cuda'), epochs=100, gpu='0', input_size=7, lr=0.001, model='lstm', n_output=7)
Epoch [1/100], Train Loss: 8.6472 Epoch [1/100], Val Loss: 8.4460
inf -> 8.446037318971422 Saving model...
Epoch [2/100], Train Loss: 8.4999 Epoch [2/100], Val Loss: 8.4308
8.446037318971422 -> 8.430825463047734 Saving model...
Epoch [3/100], Train Loss: 8.4935 Epoch [3/100], Val Loss: 8.4192
8.430825463047734 -> 8.419228898154365 Saving model...
Epoch [4/100], Train Loss: 8.4816 Epoch [4/100], Val Loss: 8.4143
8.419228898154365 -> 8.414316089064986 Saving model...
Epoch [5/100], Train Loss: 8.4697 Epoch [5/100], Val Loss: 8.4104
8.414316089064986 -> 8.410418616400825 Saving model...
Epoch [6/100], Train Loss: 8.4784 Epoch [6/100], Val Loss: 8.4168
Epoch [7/100], Train Loss: 8.4686 Epoch [7/100], Val Loss: 8.4066
8.410418616400825 -> 8.40661664362307 Saving model...
Epoch [8/100], Train Loss: 8.4749 Epoch [8/100], Val Loss: 8.4089
Epoch [9/100], Train Loss: 8.4757 Epoch [9/100], Val Loss: 8.4056
8.40661664362307 -> 8.40558934211731 Saving model...
Epoch [10/100], Train Loss: 8.4695 Epoch [10/100], Val Loss: 8.4039
8.40558934211731 -> 8.403901788923475 Saving model...
Epoch [11/100], Train Loss: 8.4670 Epoch [11/100], Val Loss: 8.4043
Epoch [12/100], Train Loss: 8.4697 Epoch [12/100], Val Loss: 8.4025
8.403901788923475 -> 8.402461378662675 Saving model...
Epoch [13/100], Train Loss: 8.4688 Epoch [13/100], Val Loss: 8.4028
Epoch [14/100], Train Loss: 8.4738 Epoch [14/100], Val Loss: 8.4027
Epoch [15/100], Train Loss: 8.4631 Epoch [15/100], Val Loss: 8.4020
8.402461378662675 -> 8.402003915221602 Saving model...
Epoch [16/100], Train Loss: 8.4689 Epoch [16/100], Val Loss: 8.4016
8.402003915221602 -> 8.40158270906519 Saving model...
Epoch [17/100], Train Loss: 8.4731 Epoch [17/100], Val Loss: 8.4034
Epoch [18/100], Train Loss: 8.4585 Epoch [18/100], Val Loss: 8.4013
8.40158270906519 -> 8.401258115415219 Saving model...
Epoch [19/100], Train Loss: 8.4640 Epoch [19/100], Val Loss: 8.4018
Epoch [20/100], Train Loss: 8.4623 Epoch [20/100], Val Loss: 8.4032
Epoch [21/100], Train Loss: 8.4616 Epoch [21/100], Val Loss: 8.4004
8.401258115415219 -> 8.40041556181731 Saving model...
Epoch [22/100], Train Loss: 8.4599 Epoch [22/100], Val Loss: 8.4010
Epoch [23/100], Train Loss: 8.4639 Epoch [23/100], Val Loss: 8.3991
8.40041556181731 -> 8.399097089414242 Saving model...
Epoch [24/100], Train Loss: 8.4587 Epoch [24/100], Val Loss: 8.3993
Epoch [25/100], Train Loss: 8.4625 Epoch [25/100], Val Loss: 8.4006
Epoch [26/100], Train Loss: 8.4598 Epoch [26/100], Val Loss: 8.3981
8.399097089414242 -> 8.398140969099822 Saving model...
Epoch [27/100], Train Loss: 8.4583 Epoch [27/100], Val Loss: 8.4004
Epoch [28/100], Train Loss: 8.4569 Epoch [28/100], Val Loss: 8.3974
8.398140969099822 -> 8.39739380059419 Saving model...
Epoch [29/100], Train Loss: 8.4634 Epoch [29/100], Val Loss: 8.3972
8.39739380059419 -> 8.397153236247876 Saving model...
Epoch [30/100], Train Loss: 8.4623 Epoch [30/100], Val Loss: 8.3974
Epoch [31/100], Train Loss: 8.4619 Epoch [31/100], Val Loss: 8.3975
Epoch [32/100], Train Loss: 8.4637 Epoch [32/100], Val Loss: 8.3971
8.397153236247876 -> 8.397107424559417 Saving model...
Epoch [33/100], Train Loss: 8.4536 Epoch [33/100], Val Loss: 8.3968
8.397107424559417 -> 8.396832130573413 Saving model...
Epoch [34/100], Train Loss: 8.4612 Epoch [34/100], Val Loss: 8.3972
Epoch [35/100], Train Loss: 8.4612 Epoch [35/100], Val Loss: 8.3962
8.396832130573413 -> 8.39616764916314 Saving model...
Epoch [36/100], Train Loss: 8.4639 Epoch [36/100], Val Loss: 8.3959
8.39616764916314 -> 8.395859365110043 Saving model...
Epoch [37/100], Train Loss: 8.4587 Epoch [37/100], Val Loss: 8.3963
Epoch [38/100], Train Loss: 8.4596 Epoch [38/100], Val Loss: 8.3962
Epoch [39/100], Train Loss: 8.4579 Epoch [39/100], Val Loss: 8.3960
Epoch [40/100], Train Loss: 8.4730 Epoch [40/100], Val Loss: 8.4017
Epoch [41/100], Train Loss: 8.4616 Epoch [41/100], Val Loss: 8.3986
Epoch [42/100], Train Loss: 8.4707 Epoch [42/100], Val Loss: 8.4047
Epoch [43/100], Train Loss: 8.4698 Epoch [43/100], Val Loss: 8.4018
Epoch [44/100], Train Loss: 8.4700 Epoch [44/100], Val Loss: 8.3990
Epoch [45/100], Train Loss: 8.4632 Epoch [45/100], Val Loss: 8.3986
Epoch [46/100], Train Loss: 8.4610 Epoch [46/100], Val Loss: 8.3971
Epoch [47/100], Train Loss: 8.4638 Epoch [47/100], Val Loss: 8.3964
Epoch [48/100], Train Loss: 8.4665 Epoch [48/100], Val Loss: 8.3948
8.395859365110043 -> 8.39482460198579 Saving model...
Epoch [49/100], Train Loss: 8.4665 Epoch [49/100], Val Loss: 8.4051
Epoch [50/100], Train Loss: 8.4630 Epoch [50/100], Val Loss: 8.4193
Epoch [51/100], Train Loss: 8.4588 Epoch [51/100], Val Loss: 8.3944
8.39482460198579 -> 8.39443603268376 Saving model...
Epoch [52/100], Train Loss: 8.4690 Epoch [52/100], Val Loss: 8.3937
8.39443603268376 -> 8.393683592478434 Saving model...
Epoch [53/100], Train Loss: 8.4631 Epoch [53/100], Val Loss: 8.3934
8.393683592478434 -> 8.393368182358918 Saving model...
Epoch [54/100], Train Loss: 8.4528 Epoch [54/100], Val Loss: 8.3964
Epoch [55/100], Train Loss: 8.4597 Epoch [55/100], Val Loss: 8.3935
Epoch [56/100], Train Loss: 8.4640 Epoch [56/100], Val Loss: 8.3991
Epoch [57/100], Train Loss: 8.4736 Epoch [57/100], Val Loss: 8.3947
Epoch [58/100], Train Loss: 8.4602 Epoch [58/100], Val Loss: 8.3940
Epoch [59/100], Train Loss: 8.4557 Epoch [59/100], Val Loss: 8.3928
8.393368182358918 -> 8.39278667944449 Saving model...
Epoch [60/100], Train Loss: 8.4518 Epoch [60/100], Val Loss: 8.3969
Epoch [61/100], Train Loss: 8.4564 Epoch [61/100], Val Loss: 8.3942
Epoch [62/100], Train Loss: 8.4606 Epoch [62/100], Val Loss: 8.3949
Epoch [63/100], Train Loss: 8.4752 Epoch [63/100], Val Loss: 8.4250
Epoch [64/100], Train Loss: 8.4689 Epoch [64/100], Val Loss: 8.3972
Epoch [65/100], Train Loss: 8.4580 Epoch [65/100], Val Loss: 8.3944
Epoch [66/100], Train Loss: 8.4606 Epoch [66/100], Val Loss: 8.3943
Epoch [67/100], Train Loss: 8.4556 Epoch [67/100], Val Loss: 8.3934
Epoch [68/100], Train Loss: 8.4582 Epoch [68/100], Val Loss: 8.3930
Epoch [69/100], Train Loss: 8.4527 Epoch [69/100], Val Loss: 8.3925
8.39278667944449 -> 8.392519588823673 Saving model...
Epoch [70/100], Train Loss: 8.4604 Epoch [70/100], Val Loss: 8.3929
Epoch [71/100], Train Loss: 8.4524 Epoch [71/100], Val Loss: 8.3929
Epoch [72/100], Train Loss: 8.4648 Epoch [72/100], Val Loss: 8.3926
Epoch [73/100], Train Loss: 8.4541 Epoch [73/100], Val Loss: 8.3943
Epoch [74/100], Train Loss: 8.4561 Epoch [74/100], Val Loss: 8.3929
Epoch [75/100], Train Loss: 8.4522 Epoch [75/100], Val Loss: 8.3931
Epoch [76/100], Train Loss: 8.4577 Epoch [76/100], Val Loss: 8.4033
Epoch [77/100], Train Loss: 8.4596 Epoch [77/100], Val Loss: 8.3987
Epoch [78/100], Train Loss: 8.4597 Epoch [78/100], Val Loss: 8.4061
Epoch [79/100], Train Loss: 8.4644 Epoch [79/100], Val Loss: 8.4047
Epoch [80/100], Train Loss: 8.4611 Epoch [80/100], Val Loss: 8.3947
Epoch [81/100], Train Loss: 8.4628 Epoch [81/100], Val Loss: 8.3993
Epoch [82/100], Train Loss: 8.4600 Epoch [82/100], Val Loss: 8.3930
Epoch [83/100], Train Loss: 8.4625 Epoch [83/100], Val Loss: 8.3918
8.392519588823673 -> 8.39178310500251 Saving model...
Epoch [84/100], Train Loss: 8.4598 Epoch [84/100], Val Loss: 8.3914
8.39178310500251 -> 8.39137864112854 Saving model...
Epoch [85/100], Train Loss: 8.4500 Epoch [85/100], Val Loss: 8.3976
Epoch [86/100], Train Loss: 8.4579 Epoch [86/100], Val Loss: 8.3952
Epoch [87/100], Train Loss: 8.4593 Epoch [87/100], Val Loss: 8.3991
Epoch [88/100], Train Loss: 8.4602 Epoch [88/100], Val Loss: 8.3930
Epoch [89/100], Train Loss: 8.4639 Epoch [89/100], Val Loss: 8.4120
Epoch [90/100], Train Loss: 8.4547 Epoch [90/100], Val Loss: 8.3935
Epoch [91/100], Train Loss: 8.4566 Epoch [91/100], Val Loss: 8.3943
Epoch [92/100], Train Loss: 8.4591 Epoch [92/100], Val Loss: 8.3930
Epoch [93/100], Train Loss: 8.4558 Epoch [93/100], Val Loss: 8.3914
Epoch [94/100], Train Loss: 8.4566 Epoch [94/100], Val Loss: 8.3909
8.39137864112854 -> 8.390944551538539 Saving model...
Epoch [95/100], Train Loss: 8.4507 Epoch [95/100], Val Loss: 8.3914
Epoch [96/100], Train Loss: 8.4551 Epoch [96/100], Val Loss: 8.3934
Epoch [97/100], Train Loss: 8.4573 Epoch [97/100], Val Loss: 8.3919
Epoch [98/100], Train Loss: 8.4487 Epoch [98/100], Val Loss: 8.3913
Epoch [99/100], Train Loss: 8.4575 Epoch [99/100], Val Loss: 8.3908
8.390944551538539 -> 8.390768245414451 Saving model...
Epoch [100/100], Train Loss: 8.4586 Epoch [100/100], Val Loss: 8.3938
Test MAE: 0.7379, Test MSE: 8.2527

Process finished with exit code 0
```
2. 对数据进行归一化后
```text
/home/mysong/.conda/envs/torch2/bin/python3 /tmp/pycharm_project_400/main.py 
Namespace(O=96, batch_size=64, dev=device(type='cuda'), epochs=100, gpu='0', input_size=7, lr=0.001, model='lstm', n_output=7)
Epoch [1/100], Train Loss: 0.0233 Epoch [1/100], Val Loss: 0.0193
inf -> 0.01925833174889838 Saving model...
Epoch [2/100], Train Loss: 0.0192 Epoch [2/100], Val Loss: 0.0191
0.01925833174889838 -> 0.019086929934996145 Saving model...
Epoch [3/100], Train Loss: 0.0190 Epoch [3/100], Val Loss: 0.0190
0.019086929934996145 -> 0.01899161700297285 Saving model...
Epoch [4/100], Train Loss: 0.0190 Epoch [4/100], Val Loss: 0.0189
0.01899161700297285 -> 0.018919970150347108 Saving model...
Epoch [5/100], Train Loss: 0.0189 Epoch [5/100], Val Loss: 0.0189
0.018919970150347108 -> 0.018864046230360313 Saving model...
Epoch [6/100], Train Loss: 0.0188 Epoch [6/100], Val Loss: 0.0188
0.018864046230360313 -> 0.01877487240428174 Saving model...
Epoch [7/100], Train Loss: 0.0187 Epoch [7/100], Val Loss: 0.0187
0.01877487240428174 -> 0.018711401391084546 Saving model...
Epoch [8/100], Train Loss: 0.0187 Epoch [8/100], Val Loss: 0.0187
0.018711401391084546 -> 0.01868720273314803 Saving model...
Epoch [9/100], Train Loss: 0.0187 Epoch [9/100], Val Loss: 0.0187
0.01868720273314803 -> 0.018663680629321822 Saving model...
Epoch [10/100], Train Loss: 0.0186 Epoch [10/100], Val Loss: 0.0186
0.018663680629321822 -> 0.01863196557732644 Saving model...
Epoch [11/100], Train Loss: 0.0186 Epoch [11/100], Val Loss: 0.0186
Epoch [12/100], Train Loss: 0.0186 Epoch [12/100], Val Loss: 0.0186
0.01863196557732644 -> 0.018605056805191218 Saving model...
Epoch [13/100], Train Loss: 0.0186 Epoch [13/100], Val Loss: 0.0186
0.018605056805191218 -> 0.01859594129577831 Saving model...
Epoch [14/100], Train Loss: 0.0186 Epoch [14/100], Val Loss: 0.0186
0.01859594129577831 -> 0.018591025257828058 Saving model...
Epoch [15/100], Train Loss: 0.0186 Epoch [15/100], Val Loss: 0.0186
0.018591025257828058 -> 0.018580493103298876 Saving model...
Epoch [16/100], Train Loss: 0.0186 Epoch [16/100], Val Loss: 0.0186
0.018580493103298876 -> 0.018567643887190906 Saving model...
Epoch [17/100], Train Loss: 0.0186 Epoch [17/100], Val Loss: 0.0186
0.018567643887190906 -> 0.018566174501622165 Saving model...
Epoch [18/100], Train Loss: 0.0186 Epoch [18/100], Val Loss: 0.0186
0.018566174501622165 -> 0.018558449525800016 Saving model...
Epoch [19/100], Train Loss: 0.0186 Epoch [19/100], Val Loss: 0.0186
0.018558449525800016 -> 0.018557480881335558 Saving model...
Epoch [20/100], Train Loss: 0.0186 Epoch [20/100], Val Loss: 0.0186
Epoch [21/100], Train Loss: 0.0185 Epoch [21/100], Val Loss: 0.0185
0.018557480881335558 -> 0.018546668950606277 Saving model...
Epoch [22/100], Train Loss: 0.0185 Epoch [22/100], Val Loss: 0.0185
0.018546668950606277 -> 0.01854131505307224 Saving model...
Epoch [23/100], Train Loss: 0.0185 Epoch [23/100], Val Loss: 0.0185
0.01854131505307224 -> 0.018536723908726817 Saving model...
Epoch [24/100], Train Loss: 0.0185 Epoch [24/100], Val Loss: 0.0185
0.018536723908726817 -> 0.018536211232896203 Saving model...
Epoch [25/100], Train Loss: 0.0185 Epoch [25/100], Val Loss: 0.0185
Epoch [26/100], Train Loss: 0.0185 Epoch [26/100], Val Loss: 0.0185
Epoch [27/100], Train Loss: 0.0185 Epoch [27/100], Val Loss: 0.0185
0.018536211232896203 -> 0.018529802629793132 Saving model...
Epoch [28/100], Train Loss: 0.0185 Epoch [28/100], Val Loss: 0.0185
Epoch [29/100], Train Loss: 0.0185 Epoch [29/100], Val Loss: 0.0185
0.018529802629793132 -> 0.01852578090296851 Saving model...
Epoch [30/100], Train Loss: 0.0185 Epoch [30/100], Val Loss: 0.0185
Epoch [31/100], Train Loss: 0.0185 Epoch [31/100], Val Loss: 0.0185
Epoch [32/100], Train Loss: 0.0185 Epoch [32/100], Val Loss: 0.0185
Epoch [33/100], Train Loss: 0.0185 Epoch [33/100], Val Loss: 0.0185
0.01852578090296851 -> 0.01852180108565975 Saving model...
Epoch [34/100], Train Loss: 0.0185 Epoch [34/100], Val Loss: 0.0185
0.01852180108565975 -> 0.018514775843532 Saving model...
Epoch [35/100], Train Loss: 0.0185 Epoch [35/100], Val Loss: 0.0185
Epoch [36/100], Train Loss: 0.0185 Epoch [36/100], Val Loss: 0.0185
Epoch [37/100], Train Loss: 0.0185 Epoch [37/100], Val Loss: 0.0185
0.018514775843532 -> 0.018503196434014373 Saving model...
Epoch [38/100], Train Loss: 0.0185 Epoch [38/100], Val Loss: 0.0185
Epoch [39/100], Train Loss: 0.0185 Epoch [39/100], Val Loss: 0.0185
Epoch [40/100], Train Loss: 0.0185 Epoch [40/100], Val Loss: 0.0186
Epoch [41/100], Train Loss: 0.0185 Epoch [41/100], Val Loss: 0.0185
0.018503196434014373 -> 0.01850210506193064 Saving model...
Epoch [42/100], Train Loss: 0.0185 Epoch [42/100], Val Loss: 0.0185
Epoch [43/100], Train Loss: 0.0185 Epoch [43/100], Val Loss: 0.0185
0.01850210506193064 -> 0.01849276486232325 Saving model...
Epoch [44/100], Train Loss: 0.0185 Epoch [44/100], Val Loss: 0.0185
Epoch [45/100], Train Loss: 0.0185 Epoch [45/100], Val Loss: 0.0185
Epoch [46/100], Train Loss: 0.0185 Epoch [46/100], Val Loss: 0.0185
0.01849276486232325 -> 0.018492748891865765 Saving model...
Epoch [47/100], Train Loss: 0.0185 Epoch [47/100], Val Loss: 0.0185
Epoch [48/100], Train Loss: 0.0185 Epoch [48/100], Val Loss: 0.0185
0.018492748891865765 -> 0.018490145465842 Saving model...
Epoch [49/100], Train Loss: 0.0185 Epoch [49/100], Val Loss: 0.0185
Epoch [50/100], Train Loss: 0.0185 Epoch [50/100], Val Loss: 0.0185
0.018490145465842 -> 0.0184854152439921 Saving model...
Epoch [51/100], Train Loss: 0.0185 Epoch [51/100], Val Loss: 0.0185
Epoch [52/100], Train Loss: 0.0185 Epoch [52/100], Val Loss: 0.0185
Epoch [53/100], Train Loss: 0.0185 Epoch [53/100], Val Loss: 0.0185
Epoch [54/100], Train Loss: 0.0185 Epoch [54/100], Val Loss: 0.0185
Epoch [55/100], Train Loss: 0.0185 Epoch [55/100], Val Loss: 0.0185
Epoch [56/100], Train Loss: 0.0185 Epoch [56/100], Val Loss: 0.0185
0.0184854152439921 -> 0.01848338123548914 Saving model...
Epoch [57/100], Train Loss: 0.0185 Epoch [57/100], Val Loss: 0.0185
Epoch [58/100], Train Loss: 0.0185 Epoch [58/100], Val Loss: 0.0185
0.01848338123548914 -> 0.01847782341280469 Saving model...
Epoch [59/100], Train Loss: 0.0185 Epoch [59/100], Val Loss: 0.0185
Epoch [60/100], Train Loss: 0.0185 Epoch [60/100], Val Loss: 0.0185
0.01847782341280469 -> 0.0184756881315951 Saving model...
Epoch [61/100], Train Loss: 0.0185 Epoch [61/100], Val Loss: 0.0185
Epoch [62/100], Train Loss: 0.0185 Epoch [62/100], Val Loss: 0.0185
Epoch [63/100], Train Loss: 0.0185 Epoch [63/100], Val Loss: 0.0185
Epoch [64/100], Train Loss: 0.0185 Epoch [64/100], Val Loss: 0.0185
Epoch [65/100], Train Loss: 0.0185 Epoch [65/100], Val Loss: 0.0185
Epoch [66/100], Train Loss: 0.0185 Epoch [66/100], Val Loss: 0.0185
0.0184756881315951 -> 0.018469800413758668 Saving model...
Epoch [67/100], Train Loss: 0.0185 Epoch [67/100], Val Loss: 0.0185
Epoch [68/100], Train Loss: 0.0185 Epoch [68/100], Val Loss: 0.0185
Epoch [69/100], Train Loss: 0.0185 Epoch [69/100], Val Loss: 0.0185
Epoch [70/100], Train Loss: 0.0186 Epoch [70/100], Val Loss: 0.0185
Epoch [71/100], Train Loss: 0.0185 Epoch [71/100], Val Loss: 0.0185
Epoch [72/100], Train Loss: 0.0185 Epoch [72/100], Val Loss: 0.0185
Epoch [73/100], Train Loss: 0.0185 Epoch [73/100], Val Loss: 0.0185
Epoch [74/100], Train Loss: 0.0185 Epoch [74/100], Val Loss: 0.0185
Epoch [75/100], Train Loss: 0.0185 Epoch [75/100], Val Loss: 0.0185
Epoch [76/100], Train Loss: 0.0185 Epoch [76/100], Val Loss: 0.0185
Epoch [77/100], Train Loss: 0.0185 Epoch [77/100], Val Loss: 0.0185
Epoch [78/100], Train Loss: 0.0185 Epoch [78/100], Val Loss: 0.0185
0.018469800413758668 -> 0.018468922211064234 Saving model...
Epoch [79/100], Train Loss: 0.0185 Epoch [79/100], Val Loss: 0.0185
Epoch [80/100], Train Loss: 0.0185 Epoch [80/100], Val Loss: 0.0185
Epoch [81/100], Train Loss: 0.0185 Epoch [81/100], Val Loss: 0.0185
0.018468922211064234 -> 0.018464949119974067 Saving model...
Epoch [82/100], Train Loss: 0.0185 Epoch [82/100], Val Loss: 0.0185
Epoch [83/100], Train Loss: 0.0185 Epoch [83/100], Val Loss: 0.0185
Epoch [84/100], Train Loss: 0.0184 Epoch [84/100], Val Loss: 0.0185
Epoch [85/100], Train Loss: 0.0185 Epoch [85/100], Val Loss: 0.0185
Epoch [86/100], Train Loss: 0.0185 Epoch [86/100], Val Loss: 0.0185
Epoch [87/100], Train Loss: 0.0185 Epoch [87/100], Val Loss: 0.0185
0.018464949119974067 -> 0.01845941809868371 Saving model...
Epoch [88/100], Train Loss: 0.0185 Epoch [88/100], Val Loss: 0.0185
Epoch [89/100], Train Loss: 0.0184 Epoch [89/100], Val Loss: 0.0185
Epoch [90/100], Train Loss: 0.0185 Epoch [90/100], Val Loss: 0.0185
Epoch [91/100], Train Loss: 0.0185 Epoch [91/100], Val Loss: 0.0185
0.01845941809868371 -> 0.018459133976311597 Saving model...
Epoch [92/100], Train Loss: 0.0184 Epoch [92/100], Val Loss: 0.0185
Epoch [93/100], Train Loss: 0.0185 Epoch [93/100], Val Loss: 0.0185
0.018459133976311597 -> 0.018453592020604346 Saving model...
Epoch [94/100], Train Loss: 0.0185 Epoch [94/100], Val Loss: 0.0185
Epoch [95/100], Train Loss: 0.0184 Epoch [95/100], Val Loss: 0.0185
Epoch [96/100], Train Loss: 0.0184 Epoch [96/100], Val Loss: 0.0185
Epoch [97/100], Train Loss: 0.0184 Epoch [97/100], Val Loss: 0.0185
Epoch [98/100], Train Loss: 0.0185 Epoch [98/100], Val Loss: 0.0185
Epoch [99/100], Train Loss: 0.0185 Epoch [99/100], Val Loss: 0.0185
Epoch [100/100], Train Loss: 0.0184 Epoch [100/100], Val Loss: 0.0185
Test MAE: 0.0430, Test MSE: 0.0184

Process finished with exit code 0
```