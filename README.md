# dl_transfer_learning
该代码主要是使用迁移学习对cifar10数据集进行分类
该代码是在pytorch框架下编写完成，迁移的预训练模型是VGG16，采用了VGG16的参数
数据集直接使用pytorch相对应的函数直接下载cifar10
base_model中主要是自己创建的模型，将两种方法进行对比，迁移学习的模型，性能更好
