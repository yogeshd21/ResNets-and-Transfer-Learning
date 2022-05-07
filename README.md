# ResNets-and-Transfer-Learning
Training different ResNet Architectures on CIFAR-10 and CIFAR-100 dataset to analyze performance of different optimizers and architectures. Using transfer learning for ResNet 50.

My Test results for the CIFAR-10 dataset were as follows (CIFAR 10.cvs) and the best accuracy was 83.15%
n value	Filter Size Start	No. of Residual Blocks	Optimizer	Accuracy
2	16	3	SGD	70.78
2	16	3	ADAM	77.54
2	16	3	RMSProp	54.75
2	32	3	SGD	74.55
2	32	3	ADAM	82.29
2	32	3	RMSProp	64.91
2	64	3	SGD	79.49
2	64	3	ADAM	81.73
2	64	3	RMSProp	67.46
2	16	4	SGD	68.81
2	16	4	ADAM	78.43
2	16	4	RMSProp	50.35
2	32	4	SGD	67.11
2	32	4	ADAM	81.75
2	32	4	RMSProp	64.37
2	64	4	SGD	76.75
2	64	4	ADAM	81.49
2	64	4	RMSProp	64.16
3	16	3	SGD	71.51
3	16	3	ADAM	75.37
3	16	3	RMSProp	46.24
3	32	3	SGD	73.22
3	32	3	ADAM	80.6
3	32	3	RMSProp	65.55
3	64	3	SGD	80.69
3	64	3	ADAM	81.88
3	64	3	RMSProp	61.89
3	16	4	SGD	68.67
3	16	4	ADAM	79.98
3	16	4	RMSProp	62.15
3	32	4	SGD	71
3	32	4	ADAM	81.06
3	32	4	RMSProp	61.61
3	64	4	SGD	75.85
3	64	4	ADAM	83.15
3	64	4	RMSProp	57.54
![image](https://user-images.githubusercontent.com/83297868/167275277-9eb41f03-d1e2-4375-b7d0-562aea714970.png)
