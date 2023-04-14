# AutoML 开发者使用手册

相比于 huggingface 等平台的预训练模型，AlphaMed 平台上的预训练模型概念更加丰富、功能更为强大。在设计一个 AlphaMed 平台的预训练模型之前，清楚二者之间的区别是非常有必要的：
- AlphaMed 平台是以联邦学习场景为主的产品，而传统预训练模型多运行在本地（集群）环境。
- AlphaMed 平台上的预训练模型预置了必要的微调参数，可在平台上直接执行训练任务，而不需要额外编码运行。
- AlphaMed 平台上的预训练模型支持在平台环境中直接部署，而不需要额外编码执行部署。

*为方便讨论，后续如无特殊说明，预训练模型均指代 AlphaMed 平台的预训练模型。*

# 设计自己的预训练模型

预训练模型主要分为三个核心组成部分：Model Card、config 配置文件、预训练模型代码，下面分别介绍。

***TODO: 补充文件系统结构示例。***

## Model Card

Model Card 提供了用于描述模型类型、使用场景、详细介绍、测试指标、使用方法等说明信息，方便模型的使用者了解、选择、使用合适的模型，助力自己的业务。Model Card 的内容主要分为两个部分：
- 格式严谨的文档标签信息；
- 格式灵活、形式丰富的模型描述信息。

Model Card 必须定义在 README.md 文件中，且此文件必须放置在预训练模型文件系统的根目录下。

### Model Card 文档标签信息

文档标签信息主要用于描述模型类型、使用场景、核心指标等基本信息，方便使用者快速了解模型的概要信息，方便平台对模型的组织和管理。模型开发者应当以严格、规范的格式提交文档标签信息，否则可能影响自己的模型在平台上的展示，给用户的使用造成不必要的麻烦。

文档标签信息必须放置在 README.md 文件的起始位置，以三个连续的“-”符号加一个换行符作为开始标识，以三个连续的“-”符号加一个换行符作为结束标识。文档标签信息采用 YAML 格式定义。以下是一个文档标签信息的示例：

```YAML
name: skin_lesion_diagnosis_avg_fed
display-name: Skin Lesion Diagnosis - FedAvg
tasks:
- image-classification
license: Apache License 2.0
framework: PyTorch
tags:
- vision
- medical
- image-classification
author: Alphamed
is-federated: true
datasets:
- Skin cancer HAM10000
```

目前平台支持的标签及说明如下：

```YAML
# 模型名称，模型名称和版本共同组成了模型的唯一标识符，必须填写
name: MODEL-NAME

# 便于对外展示的模型名称，用于概要信息、列表选择等场景，可选，默认与 name 值相同
display-name: 根据胸部 CT 成像诊断新冠病毒感染 - 同构数据联邦学习模型

# 模型任务类型，可提供多个值。仅接受下方列表中列出的值，用于模型选择过滤。必须填写
tasks:
- image-classification  # 图片分类

# 开源许可证类型，必须填写
license: Apache License 2.0

# 模型使用的 AI 框架，目前仅支持 PyTorch，可选，默认为空
framework: PyTorch

# 模型标签，可用于 UI 展示，用户自定义，可选，默认为空
tags:
- vision
- image-classification

# 模型作者，必须填写
author: AlphaMed Team

# 是否为联邦模型，可选，默认为 false
is-federated: true
```

除以上所列标签外，其它标签均会被平台忽略。

### Model Card 模型描述信息

模型描述信息用于详细描述模型细节、背景、使用方法等各种需要展示的信息。模型描述信息只用于向使用者展示，方便其了解模型的细节信息，平台不会针对模型描述信息做任何额外的处理。

模型描述信息放置在文档标签信息之后，可以在这里放置任何 Markdown 支持的格式内容，以更丰富的形式展示模型，吸引更多的使用者。

模型描述信息的内容可以参考：[Mitchell, 2018](https://arxiv.org/abs/1810.03993)

## Config 配置文件

Config 配置文件主要用于支持预训练模型的运行，主要定义预训练模型启动加载、训练、推理时需要用到的参数，比如标签、学习率、隐向量维度等。出定义平台要求的配置文件外，还可以依据业务场景和需要定义更多自定义的配置文件。

针对 AlphaMed 平台预训练模型业务场景（联邦学习）和使用对象（包含非技术人员）的特点，配置文件除需要处理核心模型本身的加载和执行逻辑外，还需要支持以下两个核心功能：
1. 支持联邦学习模型在 AlphaMed 联邦环境中的运行；
2. 支持无技术人员干预的自动化加载、运行。

因此，模型的 Config 配置文件除定义一般预训练模型常见参数外，还需要额外支持：
1. 与联邦学习环境相关的一些配置项，比如部分联邦学习算法中使用的超参数等；
2. 为支持自动加载和运行机制而设立的一些强制性规范。

### 模型配置文件

模型配置文件是平台强制要求的配置文件，必须命名为 `config.json`，必须采用 JSON 格式定义内容，且必须放置在预训练模型文件系统的根目录下。模型配置文件中必须包含平台强制要求的配置项，除此之外，可以随意定义模型需要的其它配置项。

以下是平台强制要求的配置项列表及其说明：

```JSON
{
    #####
    # entry_file 和 entry_module 定义加载 AutoModel 的入口。如果 AutoModel
    # 的完整代码在一个单独的 Python 文件中，则使用 entry_file 指定该文件在模型
    # 文件系统中的相对路径；如果 AutoModel 的定义使用了多个文件，则需要将所有定义
    # 文件统一放置在一个 Python 模块中，通过 entry_module 指定加载模型类的模块
    # 名称以载入 AutoModel 类。
    # entry_file 和 entry_module 必须且只能提供一个。
    #####
    "entry_file": "my_model.py",
    "entry_module": "my.model",
    
    # entry_class 指定预训练模型类的类名，必须提供。
    "entry_class": "MyAutoNet",
    
    # param_file 指定预训练模型类的参数文件名，必须提供。
    "param_file": "my_model.pt",
}
```

以下是一些常用配置项的示例。平台不会检查这些配置项，也不会对其附加任何约束，其定义、处理均由预训练模型的相关代码逻辑负责。尽管如此，建议在定义模型配置项时尽可能参考采用业界最常用的形式，它们往往代表了业界的最佳实践，也最有利于开发者之间的交流。

```JSON
{
    "image_size": 224,
    "torch_dtype": "float32",
    "id2label": {
        "0": "Label A",
        "1": "Label B",
        "2": "Label C"
    },
    "label2id": {
        "Label A": 0,
        "Label B": 1,
        "Label C": 2
    },
    "model_type": "vit",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_sizes": [
        32,
        64,
        160,
        256
    ],
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "num_channels": 3,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "patch_size": 16,
    "patch_sizes": [
        7,
        3,
        3,
        3
    ],
    "embedding_size": 64,
    "attention_probs_dropout_prob": 0.0,
    "use_auxiliary_head": true,
    "auxiliary_channels": 256,
    "auxiliary_loss_weight": 0.4,
    "drop_path_rate": 0.1,
}
```

### 预处理器配置文件

预处理器用于将原始数据转换为适合输入模型的格式，比如将原始图片文件转换为适配模型的张量。预处理器配置文件用于定义预处理器加载、使用的配置项，比如输入图像的分辨率、训练时的数据增强配置等。

预处理器配置文件并非必须，但如果模型需要处理原始数据，建议尽可能使用预处理器配置文件管理配置项，以更方便更灵活的支持参数更新。通常情况下预处理器配置文件都命名为 `preprocessor_config.json`，使用 JSON 格式定义内容。

以下是一些常用预处理器配置项的示例。平台不会检查这些配置项，也不会对其附加任何约束，其定义、处理均由预训练模型的相关代码逻辑负责。尽管如此，建议在定义模型配置项时尽可能参考采用业界最常用的形式，它们往往代表了业界的最佳实践，也最有利于开发者之间的交流。

1. 图片预处理器配置项
```JSON
{
    "size": 256,
    "do_center_crop": true,
    "crop_size": 224,
    "crop_pct": 0.875,
    "do_normalize": true,
    "do_resize": true,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "resample": 3,
}
```

## 预训练模型代码实现

以下是实现一个 FedAvg 联邦学习预训练模型的示例。联邦学习预训练模型即需要支持模型本身的训练、部署，同时也需要支持 FedAvg 联邦算法。为了简化设计任务，AlphaMed 平台提供了两个公共基础类 `AutoModel` 和 `FedAvgScheduler`，分别针对模型训练、部署的逻辑和 FedAvg 联邦学习的逻辑。此外，平台还在 `AutoModel` 的基础上提供了 `AutoFedAvgModel` 基础类，已提供针对 FedAvg 联邦算法环境的更丰富的支持，进一步简化 FedAvg 联邦学习预训练模型的设计。

在正式开始设计预训练模型之前，还有一些准备工作。这些准备工作不是必须的，AlphaMed 平台不会对此提出强制要求，但这些工作能够极大的帮助和促进预训练模型的设计工作。示例为模型开发者们提供了一个参考，非常建议模型开发者们根据自身需要设计好自己的辅助工具。

### 准备工作

#### 选择执行数据计算的核心模型

示例任务是根据皮肤病变部位的局部照片，判断皮肤病变类型，因此是一个图像分类任务。示例选择了经典的 ResNet18 网络作为核心模型来处理这个计算任务。核心模型本身没有什么特殊之处，这里就不再赘述了，完整的代码可以在[这里](res/res_net.py)查看。

#### 输入数据预处理

由于输入是图片数据，在传入模型之前需要将其转化为张量数据。为了方便处理，示例定义了一个 `ResNetPreprocessor` 预处理类，其不仅能够将输入的图片转化为模型接受的张量，还可以在训练时提供数据增强的支持。`ResNetPreprocessor` 继承了 AlphaMed 平台的 `Preprocessor` 基础类，其只有一个需要实现的接口：

```Python
@abstractmethod
def transform(self, image_file: str) -> torch.Tensor:
    """Transform an image object into an input tensor."""
```

transform 接口用于将图片文件转化为张量，继承 `Preprocessor` 基础类并实现这个接口可以使预处理器代码在 AlphaMed 平台上具备更好的复用性，建议尽量继承。以下为 `ResNetPreprocessor` 预处理类的完整代码：

```Python
from dataclasses import dataclass
from typing import List, Tuple

import torch
from alphafed.auto_ml.auto_model import DatasetMode, Preprocessor
from PIL import Image
from torchvision import transforms


@dataclass
class PreprocessorConfig:

    size: int
    do_affine: bool
    degrees: int
    translate: Tuple[float]
    do_horizontal_flip: bool
    image_mean: List[float]
    image_std: List[float]

    def __post_init__(self):
        self.translate = tuple(self.translate)


class ResNetPreprocessor(Preprocessor):

    def __init__(self, mode: DatasetMode, config: PreprocessorConfig) -> None:
        layers = []
        layers.append(transforms.Resize((config.size, config.size)))
        if mode == DatasetMode.TRAINING:
            if config.do_affine:
                layers.append(transforms.RandomAffine(degrees=config.degrees,
                                                      translate=config.translate))
            if config.do_horizontal_flip:
                layers.append(transforms.RandomHorizontalFlip())
        layers.append(transforms.ToTensor())
        layers.append(transforms.Normalize(config.image_mean, config.image_std))
        self._transformer = transforms.Compose(layers)

    def transform(self, image_file: str) -> torch.Tensor:
        """Transform an image object into an input tensor."""
        image = Image.open(image_file).convert('RGB')
        return self._transformer(image)
```

#### 数据集工具

示例定义了一个 `ResNetDataset` 数据集工具类，其是一个 `torch.utils.data.Dataset` 的实现，用于帮助示例处理数据集，并辅助创建训练时使用的 DataLoader。实现中用到了前面定义的 `ResNetPreprocessor`，在此不再赘述；同时还用到了 `ImageAnnotationUtils`。`ImageAnnotationUtils` 是 AlphaMed 平台提供的处理数据集标注的通用工具。以下为示例代码：

```Python
import os

from alphafed.auto_ml.auto_model import DatasetMode
from alphafed.auto_ml.cvat.annotation import ImageAnnotationUtils
from alphafed.auto_ml.exceptions import ConfigError
from torch.utils.data import Dataset


class ResNetDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 annotation_file: str,
                 mode: DatasetMode,
                 config: PreprocessorConfig) -> None:
        """Init a dataset instance for ResNet auto model families.

        Args:
            image_dir:
                The directory including image files.
            annotation_file:
                The file including annotation information.
            mode:
                One of training or validation or testing.
            config:
                The configuration for the preprocessor.
        """
        super().__init__()
        if not image_dir or not isinstance(image_dir, str):
            raise ConfigError(f'Invalid image directory: {image_dir}.')
        if not annotation_file or not isinstance(annotation_file, str):
            raise ConfigError(f'Invalid annotation file path: {annotation_file}.')
        assert mode and isinstance(mode, DatasetMode), f'Invalid dataset mode: {mode}.'
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            raise ConfigError(f'{image_dir} does not exist or is not a directory.')
        if not os.path.exists(annotation_file) or not os.path.isfile(annotation_file):
            raise ConfigError(f'{annotation_file} does not exist or is not a file.')

        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transformer = ResNetPreprocessor(mode=mode, config=config)

        self.images, self.labels = ImageAnnotationUtils.parse_single_category_annotation(
            annotation_file=self.annotation_file, resource_dir=image_dir, mode=mode
        )

    def __getitem__(self, index: int):
        _item = self.images[index]
        return self.transformer(_item.image_file), _item.class_label

    def __len__(self):
        return len(self.images)
```

### 借助 AutoModel 支持预训练模型的训练、部署

AutoModel 基础类定义了预训练模型在 AlphaMed 平台上运行需要支持的功能接口。这些接口是所有预训练模型通用的，并不仅限于联邦算法模型。因此，需要先设计实现这些基本功能。

AlphaMed 平台预训练模型即需要处理必须的平台业务逻辑，又需要处理模型本身的计算任务。如果在原生模型的基础上直接添加平台逻辑，容易导致两套逻辑的纠缠和耦合，使得设计和维护工作变得混乱且复杂。所以在示例的设计中，采用装饰器模式将核心模型封装起来，通过 `self.model` 属性获得核心模型对象，控制其完成计算任务。需要提醒的是，这里就此说明只是为了方便对示例的理解并非 AlphaMed 平台的限制。示例采用的只是一种设计方案，也可以通过其它方案实现平台规定的接口功能。

下面开始介绍 AutoModel 的核心接口。

#### 切换模型的运行模式

```Python
@abstractmethod
def train(self):
    """Go into `train` mode as of torch.nn.Module."""

@abstractmethod
def eval(self):
    """Go into `eval` mode as of torch.nn.Module."""
```

train 接口和 eval 接口将预训练模型分别设置为“训练”、“推理”模式，这两种模式的功能与常见 AI 框架，如：PyTorch、TensorFlow、PaddlePaddle 等一致，在此不赘述。对于示例预训练模型而言，只需将核心模型对象设置为相应模式即可。

```Python
def train(self):
    self.model.train()

def eval(self):
    self.model.eval()
```

#### 模型推理

```Python
@abstractmethod
def forward(self, *args, **kwargs):
    """Do a forward propagation as of torch.nn.Module."""
```

forward 接口用于使用模型完成推理任务，其输入输出取决于模型本身的设计，这一点与其它 AI 框架也高度一致。在我们的示例中，模型推理时既可以接收原始图片作为输入，以方便用户获得更好的体验；同时也接受张量形式的输入，以适应更为专业的应用场景。

```Python
@overload
def forward(self, input: torch.Tensor) -> str:
    """Predict an image's tensor and give its label."""

@overload
def forward(self, input: str) -> str:
    """Predict an image defined by a file path and give its label."""

def forward(self, input):
    if not input or not isinstance(input, (str, torch.Tensor)):
        raise AutoModelError(f'Invalid input data: {input}.')
    if isinstance(input, str):
        if not os.path.isfile(input):
            raise AutoModelError(f'Cannot find or access the image file {input}.')
        preprocessor = ResNetPreprocessor(mode=DatasetMode.PREDICTING,
                                            config=self.preprocessor_config)
        input = preprocessor.transform(input)
        input.unsqueeze_(0)
    self.model.eval()
    output: torch.Tensor = self.model(input)
    predict = output.argmax(1)[0].item()
    if not self.labels:
        if not os.path.isfile(os.path.join(self.resource_dir, 'fine_tuned.meta')):
            raise AutoModel('The `fine_tuned.meta` file is required to make prediction.')
        with open(os.path.join(self.resource_dir, 'fine_tuned.meta')) as f:
            fine_tuned_json: dict = json.loads(f)
            self.labels = fine_tuned_json.get('labels')
    return self.labels[predict]
```

#### 初始化训练数据集

```Python
@abstractmethod
def init_dataset(self, dataset_dir: str) -> Tuple[bool, str]:
    """Init local dataset and report the result.

    Args:
        dataset_dir:
            The root dir of the dataset staff.

    Return:
        Tuple[is_verification_successful, the_cause_if_it_is_failed]
    """
```

init_dataset 接口用于初始化数据集环境，以方便后续启动训练任务。比如可以在 init_dataset 中读取训练集、验证集、测试集数据，并创建各自对应的 DataLoader 对象，从而使训练过程可以更加专注于计算和更新参数。以下是示例实现：

```Python
from typing import Tuple

from alphafed import logger


def init_dataset(self, dataset_dir: str) -> Tuple[bool, str]:
    self.dataset_dir = dataset_dir
    try:
        if not self._is_dataset_initialized:
            self.training_loader
            self.validation_loader
            self.testing_loader
            if not self.training_loader or not self.testing_loader:
                logger.error('Both training data and testing data are missing.')
                return False, 'Must provide train dataset and test dataset to fine tune.'
            self.labels = (self.training_loader.dataset.labels
                            if self.training_loader
                            else self.testing_loader.dataset.labels)
            self._is_dataset_initialized = True
        return True, 'Initializing dataset complete.'
    except Exception:
        logger.exception('Failed to initialize dataset.')
        return False, '初始化数据失败，请联系模型作者排查原因。'
```

示例中的 `self.training_loader`、`self.validation_loader`、`self.testing_loader` 实现借助 Python 中的 `@property` 注解实现了对其生命周期的控制。一方面简化了训练代码的逻辑，不需要再管理 DataLoader 的创建，另外避免了因为疏忽导致错误的重新初始化 DataLoader。以下是 `self.training_loader` 的示例，`self.validation_loader`、`self.testing_loader` 与之类似。

```Python
@property
def training_loader(self) -> DataLoader:
    """Return a dataloader instance of training data.

    Data augmentation is used to improve performance, so we need to generate a new dataset
    every epoch in case of training on a same dataset over and over again.
    """
    if not hasattr(self, "_training_loader"):
        self._training_loader = self._build_training_data_loader()
    return self._training_loader

def _build_training_data_loader(self) -> Optional[DataLoader]:
    dataset = ResNetDataset(image_dir=self.dataset_dir,
                            annotation_file=self.annotation_file,
                            mode=DatasetMode.TRAINING,
                            config=self.preprocessor_config)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset=dataset,
                      batch_size=self.batch_size,
                      shuffle=True)
```

#### 定义模型微调的逻辑

有了上面的工具支持，现在可以定义预训练模型的微调逻辑了。微调逻辑定义在 fine_tune 接口中，接口定义如下：

```Python
@abstractmethod
def fine_tune(self,
              id: str,
              task_id: str,
              dataset_dir: str,
              is_initiator: bool = False,
              recover: bool = False,
              **kwargs):
    """Begin to fine-tune on dataset.

    Args:
        id:
            The ID of current node.
        task_id:
            The ID of current task.
        dataset_dir:
            The root dir of the dataset staff.
        is_initiator:
            Is current node the initiator of the task.
        recover:
            Whether run as recover mode. Recover moded is used when last fine-tuning is
            failed for some reasons, and current fine-tuning attempt to continue from
            the failure point rather than from the very begining.
        kwargs:
            Other keywords for specific models.
    """
```

先展示示例代码实现：

```Python
def fine_tune(self,
              id: str,
              task_id: str,
              dataset_dir: str,
              is_initiator: bool = False):
    self.id = id
    self.task_id = task_id
    self.is_initiator = is_initiator

    is_succ, err_msg = self.init_dataset(dataset_dir)
    if not is_succ:
        raise AutoModelError(f'Failed to initialize dataset. {err_msg}')
    num_classes = (len(self.training_loader.dataset.labels)
                   if self.training_loader
                   else len(self.testing_loader.dataset.labels))
    self._replace_fc_if_diff(num_classes)

    self.config.id2label = {str(_idx): _label for _idx, _label in enumerate(self.labels)}
    self.config.label2id = {_label: _idx for _idx, _label in enumerate(self.labels)}
    self.config.label2id = dict(sorted(self.config.label2id.items()))

    is_finished = False
    self._epoch = 0
    while not is_finished:
        self._epoch += 1
        self.push_log(f'Begin training of epoch {self._epoch}.')
        self._train_an_epoch()
        self.push_log(f'Complete training of epoch {self._epoch}.')
        is_finished = self._is_finished()

    self._save_fine_tuned()
    avg_loss, correct_rate = self._run_test()
    self.push_log('\n'.join(('Testing result:',
                             f'avg_loss={avg_loss:.4f}',
                             f'correct_rate={correct_rate:.2f}')))
```

在示例实现中，`id`、`task_id`、`dataset_dir`、`is_initiator` 参数均由平台调用时传入，用于支持联邦学习场景，暂时可以忽律它们。在后续的代码逻辑中，主要依次做了三件事情：
1. 在正式开始微调前，做一些准备工作。包括：通过调用 `init_dataset` 确保完成训练数据初始化，通过调用 `_replace_fc_if_diff` 方法和修改 `config` 的标签信息，将模型的分类层适配到训练数据集的类别标签上。在此阶段具体需要做的事情依据不同的模型和任务而各有不同，开发者可根据自己的实际需要设计。
2. 通过一个 while 循环执行微调的训练过程，这个过程与传统模型的微调流程相同。
3. 完成微调好保存训练结果，并测试微调效果。

实际运行的过程中还涉及很多辅助函数，但不影响对核心流程和理念的理解。限于篇幅不在此详述，可以通过[这里](res/auto.py)查看完成代码。

至此，示例已经是一个完备的预训练模型了，可以正常运行在 AlphaMed 平台上。然而，这样的一个模型还不具备支持联邦学习的能力，只能独自运行在自身节点上。如果需要联合几个合作方一起执行联邦学习任务，还需要更多的工作，接下来会对此进行介绍。

### 补充资料：`AutoModel` 的工具函数

`AutoModel` 除定义了预训练模型的必备接口外，还提供了一些工具，以方便开发者使用。以下是目前可用的工具：

- push_log
  
  出于安全的考虑，预训练模型有其相对封闭的运行环境。因此，如果采用普通的日志记录方式，产生的日志内容会被局限在运行环境中，不会反馈到 Playground 前端。因此当代码运行出现问题时，用户在 Playground 前端很难观察到问题及其产生的原因，进而会很难排除故障。如此则会导致训练任务卡住。

  push_log 正是为了解决运行新信息向 Playground 前端传递而提供的工具。通过 push_log 发送的信息将会自动同步至 Playground 前端，从而使所有任务参与方均能够更细致的跟踪和观察任务运行状态。

  push_log 的使用方法非常简单，只需要将想要传送到 Playground 的文本消息传入即可，下面是一个示例：

  ```Python
  self.push_log('Saved latest parameters locally.')
  ```

## 实现一个支持 FedAvg 算法的联邦预训练模型

如前所述，刚才实现的 `AutoResNet` 是一个仅能在本地单独运行的预训练模型，若要支持 FedAvg 联邦学习，还需要做更多的工作。幸运的是，AlphaMed 也为此提供了一些新的工具，以帮助设计实现支持 FedAvg 算法的预训练模型。基于前面刚刚学习过的内容，下面先来介绍 `AutoFedAvgModel` 基础类。

### 借助 AutoFedAvgModel 帮助预训练模型支持 FedAvg 联邦学习算法

AutoFedAvgModel 定义了一系列接口，用于支持 FedAvg 联邦学习算法的运行。定义这些接口的目的是为了帮助开发者无须关注 FedAvg 算法的控制和运行，只需专注于模型本身需要处理的计算任务。开发者需要根据实际情况实现这些接口。无须太过担心，AutoFedAvgModel 定义的接口并不复杂，此外，AutoFedAvgModel 也依然是一个标准的预训练模型，因此大部分之前的工作都可以在此复用。下面开始正式介绍。

#### 准备微调时使用的数据集加载器

AutoFedAvgModel 定义了三个接口，分别返回微调过程中使用的训练集、验证集和测试集的数据加载器对象。

```Python
@property
@abstractmethod
def training_loader(self) -> DataLoader:
    """Return the dataloader object used in training."""

@property
@abstractmethod
def validation_loader(self) -> DataLoader:
    """Return the dataloader object used in validation."""

@property
@abstractmethod
def testing_loader(self) -> DataLoader:
    """Return the dataloader object used in testing."""
```

回头翻看 `AutoResNet` 的实现会发现，其正是按照这三个接口的要求实现的。因此建议设计者参考 `AutoResNet` 管理预训练模型的数据集，则这里不需要再做额外的工作了。如果采用了不同的设计，则需要在这里补充实现这三个接口。

#### 准备微调时使用的模型对象和优化器对象

与前面返回数据集加载器对象的接口类似，这里定义的接口分别返回的是微调过程中使用的核心模型对象和优化器对象。

```Python
@property
@abstractmethod
def model(self) -> nn.Module:
    """Return the model object used in training and predicting."""

@property
@abstractmethod
def optimizer(self) -> optim.Optimizer:
    """Return the optimizer object used in training."""
```

同样，`AutoResNet` 的实现也正是符合接口要求的，因此不需要再做额外的工作。

#### 传递和加载 FedAvg 算法需要更新的参数

下面两个接口针对的是 FedAvg 算法运行过程中聚合更新的参数。其中一个返回当前计算节点的本地参数，用于传输至其它参与方进一步处理。另一个则是使用收到的参数更新本地模型。

```Python
@abstractmethod
def state_dict(self) -> Dict[str, torch.Tensor]:
    """Get the params that need to train and update.

    Only the params returned by this function will be updated and saved during aggregation.

    Return:
        List[torch.Tensor], The list of model params.
    """

@abstractmethod
def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
    """Load the params that trained and updated.

    Only the params returned by state_dict() should be loaded by this function.
    """
```

得益于示例设计中对核心模型的封装，这两个接口的实现如下：

```Python
def state_dict(self) -> Dict[str, torch.Tensor]:
    return self.model.state_dict()

def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
    return self.model.load_state_dict(state_dict)
```

#### 定义训练、测试、验证的逻辑

首先来看如何定义训练逻辑，相关接口为 `train_an_epoch`。正如其名称所展示的，这里定义的是完成一轮训练的逻辑，而不是完整训练流程的逻辑。由于接口针对模型本身的运算和 FedAvg 算法的实现提供了隔离，所以这里所说的一轮训练仅仅是指模型在计算节点本地完成的一轮训练。既然不涉及联邦算法，这里的训练逻辑就和一个普通的、在本地运行的模型别无二致了。

```Python
@abstractmethod
def train_an_epoch(self) -> Any:
    """Define the training steps in an epoch."""
```

示例中的实现代码如下：

```Python
def _train_an_epoch(self):
    self.train()
    for images, targets in self.training_loader:
        if self.is_cuda:
            images, targets = images.cuda(), targets.cuda()
        output = self.model(images)
        output = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_an_epoch(self):
    self._epoch += 1
    self._train_an_epoch()
```

实现中的抽象拆分是为了方便兼顾本地运行模式与联邦运行模式，并非平台要求。实际设计时可依据自身需要进行修改。`train_an_epoch` 接口也可视需要返回一些结果数据，平台对此无强制要求，由设计者根据需要自行决定。

然后来看测试逻辑，相关接口为 `run_test`。与本地模型的训练一样，测试通常不区分轮次，只需要在必要时完整运行一次得到最终结果。

```Python
@abstractmethod
def run_test(self) -> Any:
    """Run a round of test."""
```

示例中的实现代码如下：

```Python
def run_test(self) -> Tuple[float, float]:
    """Run a test and report the result.

    Return:
        avg_loss, correct_rate
    """
    return self._run_test()
```

如示例中所示，`run_test` 返回了一些执行结果，以供后续逻辑继续处理。

最后是验证逻辑，相关接口为 `run_validation`。依据训练方案的各自差异，验证逻辑不是必须的。设计者可以根据自身需要选择是否提供具体实现。

```Python
def run_validation(self) -> Any:
    """Run a round of validation."""
    raise NotImplementedError()
```

#### 微调完成后需要保存的微调文件

默认情况下，在微调任务成功完成后，AlphaMed 平台会自动打包保存训练完成后的新参数（取自设计者实现的 `state_dict` 接口返回），并附带预训练模型附带的其它原始资源，比如 config.json、预训练模型定义文件（目录）等。但其中并不包括原始 config.json 文件中指定的 param_file 参数文件，因为一般情况下它已经不再有意义了。在很多情况下这些资源已足以支持后续部署微调后的新模型。但是，依据微调任务的不同，可能会存在更进一步的需求。比如，模型完成微调后，除了需要保存新的模型参数外，还需要保存针对下游任务设置的新标签信息。此时默认实现就不能满足新的需求了。

针对这种情况，AlphaMed 平台提供了一套机制，帮助模型设计者将任何需要额外保存的数据写入文件，随微调结果一起打包保存。从而在下次启动和使用微调好的新模型时，能够从相关文件中读取出任意需要的信息。

但是在使用这项特性时还有一点需要额外注意。出于安全原因，预训练模型的运行环境文件系统是收到管控的，因此保存数据的文件必须写入指定文件夹，否则将不能保证能够成功处理。这个指定的文件夹可以通过 `AutoFedAvgModel` 对象的 `result_dir` 属性获得。平台会负责管理和维护每个任务的文件夹，不可以修改 `result_dir` 的值。

按照上述要求写入需要保存的文件之后，可以通过 `fine_tuned_files_dict` 方法返回需要保存的文件以及保存的路径，平台将按照要求将这些文件添加到保存列表，与默认文件一起保存并提供下载功能。

```Python
def fine_tuned_files_dict(self) -> Optional[Dict[str, str]]:
    """Return the fine-tuned files used by reinitializing the fine-tuned model.

    The necessary information for reinitialize a fine-tuned model object
    after training, i.e. the new label list, can be saved in files whose
    path will be returned in `fine_tuned_files`. Those files will then be
    included in the result package and be persistent in the task context.
    So that it can be downloaded later and used by reinitializing the
    fine-tuned model.

    The records of files are in format:
    {
        'relative_path_to_resource_dir_root': 'real_path_to_access_file'
    }
    """
```

举例来说，假设将下游任务的新标签保存在 `result_dir/new_labels.json` 文件中，则可以让 `fine_tuned_files_dict` 返回 {'result_dir/new_labels.json': 'new_labels.json'}。微调任务成功完成后，此文件将会被保存在模型结果资源包的根目录下。如果不需要此功能，使 `fine_tuned_files_dict` 返回 None 或者空值即可。

至此，`AutoFedAvgModel` 的设计工作已经完成。了解[如何在 AlphaMed 平台上构建 FedAvg 联邦学习任务](../fed_avg/FedAvg.ipynb)的开发者们，可能已经注意到了，`AutoFedAvgModel` 的大部分设计与 `FedAvgScheduler` 的接口很像。确实如此，`AutoFedAvgModel` 的设计目标正是为了更好的配合 `FedAvgScheduler` 已完成联邦任务。所以接下来，开始设计预训练模型的 `FedAvgScheduler` 实现。

## 实现预训练模型的 FedAvgScheduler

预训练模型的 `FedAvgScheduler` 其实和动态构建的 `FedAvgScheduler` 是一样的，如果对于如何构建 FedAvg 联邦学习任务还不熟悉，建议先去这里[学习](../fed_avg/FedAvg.ipynb)了解，会更容易理解一下的说明内容。

同时，为了进一步简化预训练模型 `FedAvgScheduler` 的设计工作，平台提供了 `AutoFedAvgScheduler` 基类。`AutoFedAvgScheduler` 基类在原始 `FedAvgScheduler` 基类的基础上，针对预训练模型任务环境和业务要求做了一些实现上的更新于升级，同时保持了 `FedAvgScheduler` 基类的封装思想。使用 `AutoFedAvgScheduler` 基类可以帮助设计者专注于模型计算逻辑，避免纠缠于 FedAvg 算法和平台运行环境的细节中去。

### 在 `AutoFedAvgScheduler` 中注入 `AutoFedAvgModel` 对象

示例中已经设计了一个可以有效配合 `AutoFedAvgScheduler` 工作的 `AutoFedAvgModel`，现在需要在 `AutoFedAvgScheduler` 初始化时将其注入进去。为此 `AutoFedAvgScheduler` 在 \_\_init__ 方法中定义一个参数 `auto_proxy` 来接收 `AutoFedAvgModel` 对象。因此在设计自己的预训练模型 Scheduler 时，不要忘了这个初始化参数。

```Python
def __init__(self,
             auto_proxy: AutoFedAvgModel,
             max_rounds: int = 0,
             merge_epochs: int = 1,
             calculation_timeout: int = 300,
             schedule_timeout: int = 30,
             log_rounds: int = 0,
             involve_aggregator: bool = False,
             **kwargs):
    super().__init__(max_rounds=max_rounds,
                     merge_epochs=merge_epochs,
                     calculation_timeout=calculation_timeout,
                     schedule_timeout=schedule_timeout,
                     log_rounds=log_rounds,
                     involve_aggregator=involve_aggregator)
    self.auto_proxy = auto_proxy
```

其它参数的使用方式与标准 `FedAvgScheduler` 相同，这里不再赘述。

### 获取微调时使用到的各种资源对象

借助于 `AutoFedAvgModel` 对象的帮助，`AutoFedAvgScheduler` 获取微调训练时使用的资源对象变得非常方便。如无特殊的需要，可以采用 `AutoFedAvgScheduler` 提供的默认实现，能够满足绝大部分情况的需求。

```Python
def build_model(self) -> nn.Module:
    return self.auto_proxy.model

def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
    return self.auto_proxy.optimizer

def build_train_dataloader(self) -> DataLoader:
    return self.auto_proxy.training_loader

def build_validation_dataloader(self) -> DataLoader:
    return self.auto_proxy.validation_loader

def build_test_dataloader(self) -> DataLoader:
    return self.auto_proxy.testing_loader

def state_dict(self) -> Dict[str, torch.Tensor]:
    return self.auto_proxy.state_dict()

def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
    return self.auto_proxy.load_state_dict(state_dict)
```

### 记录目前为止效果最好的参数

最新的 `FedAvgScheduler` 体系加入了验证集的支持。在有验证集参与的学习流程中，最终需要保存的训练结果不一定是最后一轮的参数更新，而可能是若干步之前经验证集验证的某一次更新。为了支持这种情况，`AutoFedAvgScheduler` 定义了新接口 `best_state_dict` 用于返回目前为止效果最好的参数，可以通过控制这个接口的返回来决定最终采用的参数数据。

```Python
@property
@abstractmethod
def best_state_dict(self) -> Dict[str, torch.Tensor]:
    """Return the best state of the model by now."""
```

### 定义训练、测试逻辑

同样的，`AutoFedAvgModel` 中已经定义了训练、测试的逻辑，因此在 Scheduler 中可以很方便的直接使用。

```Python
def train_an_epoch(self):
    self.auto_proxy.train_an_epoch()

def test(self):
    avg_loss, correct_rate = self.auto_proxy.run_test()

    self.tb_writer.add_scalar('test_results/avg_loss', avg_loss, self.current_round)
    self.tb_writer.add_scalar('test_results/correct_rate', correct_rate, self.current_round)
```

### 重新实现 `is_task_finished`，支持早停算法

由于有了验证集的支持，可以在微调训练过程中加入早停算法，从而有效的避免过拟合。要实现此目的，需要修改 FedAvg 算法中单纯依赖训练轮次来决定是否完成训练的判断逻辑，改为由训练轮次和验证集结果共同决定是否停止训练。以下是示例中的实现：

```Python
def is_task_finished(self) -> bool:
    """Decide if stop training.

    If there are validation dataset, decide depending on validatation results. If
    the validation result of current epoch is below the best record for 10 continuous
    times, then stop training.
    If there are no validation dataset, run for `max_rounds` times.
    """
    if not self.validation_loader or len(self.validation_loader) == 0:
        self._best_state = deepcopy(self.state_dict())
        return self._is_reach_max_rounds()

    # make a validation
    self.push_log(f'Begin validation of round {self.current_round}.')
    avg_loss, correct_rate = self.auto_proxy.run_validation()
    self.push_log('\n'.join(('Validation result:',
                             f'avg_loss={avg_loss:.4f}',
                             f'correct_rate={correct_rate:.2f}')))

    if correct_rate > self._best_result:
        self._overfit_index = 0
        self._best_result = correct_rate
        self._best_state = deepcopy(self.state_dict())
        self.push_log('Validation result is better than last epoch.')
        return self._is_reach_max_rounds()
    else:
        self._overfit_index += 1
        msg = f'Validation result gets worse for {self._overfit_index} consecutive times.'
        self.push_log(msg)
        return self._overfit_index >= 10 or self._is_reach_max_rounds()
```

至此，一个可以支持 FedAvg 联邦算法的预训练模型已经准备就绪了。完整代码可以查看[这里](res/auto.py)，将其上传至 AlphaMed 平台“我的模型”后，就可以开始使用了。

# 上传预训练模型

要上传设计好的预训练模型，首先需要登录 [AlphaMed Playground 系统](http://playground.ssplabs.com/login)。如果还没有账号，需要先注册一个账号。登录成功后在“我的空间”中选择“创建”->“模型”。

![创建新模型入口](res/1.png)

在表格中填写必要的字段信息，并上传包含 Model Card 信息的 README.md 文件。

![填写创建新模型表单](res/2.png)

创建模型执行成功后，会进入“模型信息”页面，显示 Model Card 中填写的内容。如果需要修改信息，可以在这里执行编辑。

![模型信息](res/3.png)

此时模型文件系统中还只有 README.md 文件，还需要继续上传模型代码、配置、参数等其它文件。可以点击“模型信息”旁边的“模型文件”标签，上传所有相关文件。

![上传其它模型文件](res/4.png)

# ***大功告成！***
