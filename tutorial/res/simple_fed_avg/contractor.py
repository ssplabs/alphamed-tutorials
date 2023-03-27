from dataclasses import dataclass

from alphafed.contractor import (ContractEvent, TaskMessageContractor,
                                 TaskMessageEventFactory)


@dataclass
class CheckInEvent(ContractEvent):
    """集合请求消息。"""

    TYPE = 'check_in'

    peer_id: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CheckInEvent':
        event_type = contract.get('type')
        peer_id = contract.get('peer_id')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert peer_id and isinstance(peer_id, str), f'invalid peer_id: {peer_id}'
        return CheckInEvent(peer_id=peer_id)


@dataclass
class CheckInResponseEvent(ContractEvent):
    """集合响应消息。"""

    TYPE = 'check_in_resp'

    aggr_id: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CheckInResponseEvent':
        event_type = contract.get('type')
        aggr_id = contract.get('aggr_id')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert aggr_id and isinstance(aggr_id, str), f'invalid aggr_id: {aggr_id}'
        return CheckInResponseEvent(aggr_id=aggr_id)


@dataclass
class StartEvent(ContractEvent):
    """开始训练消息。"""

    TYPE = 'start'

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'StartEvent':
        event_type = contract.get('type')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        return StartEvent()


@dataclass
class CloseEvent(ContractEvent):
    """训练结束消息。"""

    TYPE = 'close'

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CloseEvent':
        event_type = contract.get('type')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        return CloseEvent()


class SimpleFedAvgEventFactory(TaskMessageEventFactory):

    _CLASS_MAP = {
        CheckInEvent.TYPE: CheckInEvent,
        CheckInResponseEvent.TYPE: CheckInResponseEvent,
        StartEvent.TYPE: StartEvent,
        CloseEvent.TYPE: CloseEvent,
        **TaskMessageEventFactory._CLASS_MAP
    }


class SimpleFedAvgContractor(TaskMessageContractor):

    def __init__(self, task_id: str):
        super().__init__(task_id=task_id)
        self._event_factory = SimpleFedAvgEventFactory

    def check_in(self, peer_id: str):
        """发送集合请求消息。"""
        event = CheckInEvent(peer_id=peer_id)
        self._new_contract(targets=self.EVERYONE, event=event)

    def response_check_in(self, aggr_id: str, peer_id: str):
        """发送集合响应消息。"""
        event = CheckInResponseEvent(aggr_id=aggr_id)
        self._new_contract(targets=[peer_id], event=event)

    def start(self):
        """发送开始训练消息。"""
        event = StartEvent()
        self._new_contract(targets=self.EVERYONE, event=event)

    def close(self):
        """发送训练结束消息。"""
        event = CloseEvent()
        self._new_contract(targets=self.EVERYONE, event=event)
