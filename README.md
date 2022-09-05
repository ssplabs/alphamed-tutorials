# WELCOME TO ALPHAMED TUTORIALS

AlphaMed 是一个基于区块链技术的去中心化联邦学习解决方案，旨在使医疗机构能够在保证其医疗数据隐私和安全的同时，实现多机构联合建模。医疗机构可以在本地节点实现模型的训练，并支持以匿名的身份将加密的参数共享至聚合节点，从而实现更安全、可信的联邦学习。

相比于传统的联邦学习，AlphaMed 平台不仅能够确保只有合法的且经过许可的参与者才能加入网络，同时支持节点的匿名化的参与联合建模。同时，区块链的共识算法能够确保网络中的节点得到一直的决策，恶意的参与者或者数据投毒等攻击将被拒绝，从而保证了联邦学习更好的安全性。

在联邦学习的过程中，各个参与方都受到智能合约的约束，并且所有的事件、操作都将被记录在区块链的分布式账本上，可追溯、可审计，使得联合机器学习的安全性和隐私保护能力极大的提升。

## 项目目标

1. 学习并实践 PyTorch 框架知识，搞懂自己分配到的 PyTorch 官方示例；
2. 参考 AlphaMed mnist.md 中的示例，将 PyTorch 的官方示例，扩展为 AlphaMed 平台上的联邦学习模型；
3. 将自己修改的模型，以联邦学习的形式，在 AlphaMed Playground 用户界面上成功运行，并获得预期的训练结果；
4. 编写文档和代码、注释，帮助其他同学掌握并复现自己的联邦学习示例。

## 预期收获

1. 熟悉 PyTorch 与深度学习；熟悉 AlphaMed 与联邦学习；
2. 能够动手编写、调试并实现 PyTorch 联邦学习模型，用于解决视觉、文本、语音等领域的机器学习任务；
3. 基于自己的实践成果，扩展或改进之后提交至大创项目。

## 操作步骤
1. 克隆本项目 git 仓库，用户获取说明、示例并提交自己的实验成果；
2. 每人在以下候选 PyTorch 官方示例中认领一个（互相不要重复）：
   - [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
   - [LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
   - [SPEECH COMMAND CLASSIFICATION WITH TORCHAUDIO](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html)
   - [TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

   在本地环境中成功运行自己选择的示例，训练模型并获得预期的结果。在学习示例的过程中，理解和掌握示例模型的核心知识点：
   1. 认真阅读模型介绍，理解模型针对的是什么问题，以及解决问题的原理和思路；
   2. 深入理解模型训练过程中的数据加载流程。学习 PyTorch 的 Dataset 和 DataLoader 机制，弄清楚原始数据是如何一步步变成输入模型的 Tensor 对象的；
   3. 掌握如何在 PyTorch 中定义模型。学会使用 torch.nn 模块中的一些常用组件，以及 torch.nn.functional 模块中的对应函数。
   4. 掌握如何在 PyTorch 中定义损失函数、优化器等组件？
   5. 掌握如何使用 PyTorch 模型做训练和推理。结合学习到的前向传播和反向传播理论，了解 PyTorch 中的自动梯度计算和参数更新机制。
3. 仔细阅读 [MNIST 联邦学习改造示例](./mnist.md) ，结合自己选择的示例模型的数据加载方式，设计编写  DatasetVerify.py 数据验证代码。
4. 仔细阅读 [MNIST 联邦学习改造示例](./mnist.md) ，参考示例，针对 AlphaMed 平台要求逐步改造自己选择的示例模型。
5. 登录 AlphaMed 平台，登录地址可以是以下四个地址中的任意一个。建议每位同学选择一个，以尽量避免互相干扰。（登录地址及用户名密码线下发给大家。）
6. 新建一个项目，填写页面表单，上传项目附件和 DatasetVerify 数据验证代码文件，选择“横向”联邦学习，选择定向节点，选中其它参与节点，创建新的任务。
7. 登录创建任务时指定的其它参与节点，完成数据验证，并加入任务。如果数据验证时出现错误，需要检查日志输出，定位错误原因并调整原始数据或验证代码，保证数据验证通过。
8. 启动训练，等待训练成功结束。如果训练期间发生了错误，需要检查日志输出，定位错误原因并修正代码，然后重新尝试直至成功完成。
9. 下载模型指标数据，验证是否符合预期。如不符合预期，需要检查日志输出，定位错误原因并修正代码，然后重新尝试。
10. 下载训练后的模型，在线下本地环境中通过 torch.jit 机制加载模型，验证模型推理结果是否符合预期。
11. 编写文档，描述自己所选模型的改造步骤和说明。将说明文档和改造模型代码上传至 git 仓库，供其它同学参考和学习。
12. \[选修\]阅读其它同学改造完成的模型代码和说明文件，在 AlphaMed 平台上复现其结果。
13. \[选修\]选择其它感兴趣的模型，尝试改造并运行在 AlphaMed 平台上。
