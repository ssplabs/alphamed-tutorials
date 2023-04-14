# 在参与方之间传递消息

对于联邦算法实现来说，所有的通信都是基于任务内相关节点之间的。当需要传输消息时，可以基于`TaskMessageContractor`机制扩展合约消息，然后将合约消息发送至接收方。`TaskMessageContractor`本身已经提供了基础的消息发送、接收、校验机制，开发者需要根据算法需要定制消息的内容。以下为定义、发送、接收一个自定义消息的示例。

## 消息结构定义

自定义的消息结构必须继承`ContractEvent`虚拟基础类。`ContractEvent`虚拟基础类定义了与平台合约消息机制互动的接口，正确实现接口后平台就可以正确管理合约消息在区块链网络上的发送、接收；同时，`ContractEvent`虚拟基础类也提供了一部分通用实现，可以降低开发负担。自定义的消息结构必须是一个`dataclass`类型，因为`ContractEvent`虚拟基础类的正常运转依赖 Python 的`dataclasses`机制。

自定义的消息结构必须定义 TYPE 属性，且要保证不同消息的 TYPE 值唯一。TYPE 值帮助消息解析时的类型判断，其值为普通字符串。

自定义的消息结构可以任意定义随消息传递的数据，比如示例中的 peer_id，nonce，也可以不定义任何附带数据。（消息类型本身即是一种信息，所以不会传输空信息。）要求消息结构中定义的所有附带数据都必须能够 JSON 化。`ContractEvent`基类支持常见基本数据类型自动 JSON 化，如果发现自定义的类型无法 JSON 化，需要重新实现`event_to_contract`接口，将其手动转化为可以 JSON 化的形式，并在`contract_to_event`中还原成原始类型返回。

自定义的消息结构必须实现`contract_to_event`接口，其作用是将收到的 JSON 字典化的消息内容，重新结构化为消息类型对象，以方便业务逻辑代码使用。一般情况下，不建议在逻辑代码中直接操作 JSON 字典数据，因为普通字典结构缺乏类型完整性约束，且无法利用 IDE 的类型检查和提示工具，很容易引入 bug。

```Python
from dataclasses import dataclass
from alphafed.contractor import ContractEvent

@dataclass
class CheckinEvent(ContractEvent):
    """An event of checkin for a specific task."""

    TYPE = 'checkin'

    peer_id: str
    nonce: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CheckinEvent':
        event_type = contract.get('type')
        peer_id = contract.get('peer_id')
        nonce = contract.get('nonce')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert peer_id and isinstance(peer_id, str), f'invalid peer_id: {peer_id}'
        assert nonce or isinstance(nonce, str), f'invalid nonce: {nonce}'
        return CheckinEvent(peer_id=peer_id, nonce=nonce)
```

## 消息类型注册

消息类型的注册借助`EventFactory`机制实现。自定义的`EventFactory`必须继承`TaskMessageEventFactory`基础类，并在`TaskMessageEventFactory`基础类包含的消息类型基础上添加注册自定义的消息类型。注册方式是在`_CLASS_MAP`类属性中添加 TYEP: EventClass 的键值对。由于联邦算法用到的消息都是在算法流程设计时可以确定的，因此一般采用静态配置完成消息注册即可。如特殊情况下需要支持动态注册，请开发者自行实现。

```Python
from alphafed.contractor import TaskMessageEventFactory

class FedAvgEventFactory(TaskMessageEventFactory):

    _CLASS_MAP = {
        CheckinEvent.TYPE: CheckinEvent,
        CheckinResponseEvent.TYPE: CheckinResponseEvent,
        SyncStateEvent.TYPE: SyncStateEvent,
        SyncStateResponseEvent.TYPE: SyncStateResponseEvent,
        StartAggregatorElectionEvent.TYPE: StartAggregatorElectionEvent,
        BenchmarkDoneEvent.TYPE: BenchmarkDoneEvent,
        CloseAggregatorElectionEvent.TYPE: CloseAggregatorElectionEvent,
        StartRoundEvent.TYPE: StartRoundEvent,
        ReadyForAggregationEvent.TYPE: ReadyForAggregationEvent,
        CloseRoundEvent.TYPE: CloseRoundEvent,
        FinishTaskEvent.TYPE: FinishTaskEvent,
        ResetRoundEvent.TYPE: ResetRoundEvent,
        **TaskMessageEventFactory._CLASS_MAP
    }
```

## 消息发送接口

请注意，此处的消息发送接口指的并不是如何将合约消息内容发送到区块链网络上的接口，这部分功能已经由平台提供的`TaskMessageContractor`基类实现，应当直接调用。此处的消息发送接口指的是面向业务逻辑，发送特定类型合约消息的接口，比如发送“开始训练”消息的接口。
消息发送接口应当集中定义在一个继承了`TaskMessageContractor`基类的消息合约类实现中，这样可以复用`TaskMessageContractor`基类提供的现有接口，减少接口开发工作量。
自定义的消息合约类需要在`__init__`初始化函数中将默认的`_event_factory`替换为上一步自定义的`EventFactory`对象，否则将不能识别自定义的消息类型。

```Python
import secrets
from alphafed.contractor import TaskMessageContractor

class FedAvgContractor(TaskMessageContractor):

    def __init__(self, task_id: str):
        super().__init__(task_id=task_id)
        self._event_factory = FedAvgEventFactory

    def checkin(self, peer_id: str) -> str:
        """Checkin to the task.

        :return
            A nonce string used for identifying matched sync_state reply.
        """
        nonce = secrets.token_hex(16)
        event = CheckinEvent(peer_id=peer_id, nonce=nonce)
        self._new_contract(targets=self.EVERYONE, event=event)
        return nonce
```

## 接收消息

接收方需要接受指定消息时，可通过`TaskMessageContractor`提供的`contract_events`接口持续监听合约消息，并在收到消息后判断消息的类型，根据需要执行后续的处理。`contract_events`接口是一个消息的迭代器，默认情况下会持续不断地监听并返回新的消息。因此实际使用中需要合理控制循环的退出，或者在调用时指定超时时间，避免陷入无限循环。

```Python
for _event in self.contractor.contract_events():
    if isinstance(_event, CheckinEvent):
        if _event.peer_id not in self._participants:
            self._participants.append(_event.peer_id)
            self.push_log(f'Welcome a new participant ID: {_event.peer_id}.')
            self.push_log(f'There are {len(self._participants)} participants now.')
        self.contractor.respond_check_in(round=self.round,
                                         aggregator=self.id,
                                         nonce=_event.nonce,
                                         requester_id=_event.peer_id)
        if len(self._participants) == self.clients:
            break  # 退出循环
```
